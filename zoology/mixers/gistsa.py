# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import wandb

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import RMSNorm, ShortConvolution
from fla.modules.feature_map import ReLUFeatureMap, SwishFeatureMap, T2RFeatureMap
from fla.modules.layernorm import rms_norm_linear
from fla.ops.gistsa import chunk_gistsa, fused_recurrent_gistsa, fused_recurrent_gistsa_auto

from zoology.logger import WandbLogger

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

    from fla.models.utils import Cache


class GistSlotAttention(nn.Module):

    def __init__(
        self,
        mode: str = 'chunk',
        d_model: int = 1024,
        expand_k: float = 1.,
        expand_v: float = 1.,
        num_heads: int = 4,
        num_kv_heads: int | None = None,
        use_short_conv: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        num_slots: int | None = None,
        elementwise_affine: bool | None = True,
        norm_eps: float = 1e-5,
        gate_logit_normalizer: int = 8,
        feature_map: str = 'swish',
        use_output_gate: bool = False,
        use_norm: bool = True,
        layer_idx: int | None = None,
        scale: float | None = 1.,
        self_proto: bool = False,
        **kwargs,
    ) -> GistSlotAttention:
        super().__init__()

        self.mode = mode
        self.d_model = d_model
        self.expand_k = expand_k
        self.expand_v = expand_v
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.key_dim = int(d_model * expand_k)
        self.value_dim = int(d_model * expand_v)
        self.key_dim_per_group = self.key_dim // self.num_kv_groups
        self.value_dim_per_group = self.value_dim // self.num_kv_groups
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.conv_bias = conv_bias

        self.gate_logit_normalizer = gate_logit_normalizer

        self.use_output_gate = use_output_gate
        self.use_norm = use_norm
        self.scale = scale

        if num_slots is None:
            num_slots = self.head_k_dim
        self.num_slots = num_slots
        if self_proto:
            self.num_slots = 1  # We are retrofitting the slot weight matrix to a binary classifier
        
        self.self_proto = self_proto

        self.layer_idx = layer_idx

        if layer_idx is None:
            warnings.warn(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class.",
            )

        self.register_module('feature_map', None)
        if feature_map == 'swish':
            self.feature_map = SwishFeatureMap()
        elif feature_map == 'relu':
            self.feature_map = ReLUFeatureMap()
        elif feature_map == 't2r':
            self.feature_map = T2RFeatureMap(self.head_k_dim, self.head_k_dim)
        else:
            raise NotImplementedError(f"Feature map `{feature_map}` is not supported now.")

        self.q_proj = nn.Linear(self.d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.key_dim_per_group, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.value_dim_per_group, bias=False)
        self.p_proj = nn.Linear(self.d_model, 1, bias=True)
        self.t_proj = nn.Linear(self.d_model, 1, bias=True)

        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                d_model=self.key_dim,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.k_conv1d = ShortConvolution(
                d_model=self.key_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                d_model=self.value_dim_per_group,
                kernel_size=conv_size,
                bias=conv_bias,
                activation='silu',
            )

        self.g_norm = RMSNorm(self.d_model, elementwise_affine, eps=norm_eps)
        self.o_proj = nn.Linear(self.value_dim, self.d_model, bias=False)


        # Variable to track p_t for sparsity regularization
        self.p = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        gist_start: int | None = None,
        n_gists: int | None = None,
        gist_idx: torch.Tensor | None = None,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.Tensor, torch.Tensor | None, Cache | None]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape
        mode = 'fused_recurrent' if hidden_states.shape[1] <= 64 else self.mode

        last_state = None
        if past_key_values is not None and len(past_key_values) > self.layer_idx:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get('cu_seqlens')
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices).unsqueeze(0)

        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state['conv_state']
            q, conv_state_q = self.q_conv1d(
                x=self.q_proj(hidden_states),
                cache=conv_state_q,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            k, conv_state_k = self.k_conv1d(
                x=self.k_proj(hidden_states),
                cache=conv_state_k,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        p = self.p_proj(hidden_states)
        t = self.t_proj(hidden_states)
        t = F.sigmoid(t)

        q = rearrange(q, '... (h d) -> ... h d', d=self.head_k_dim)
        k = rearrange(k, '... (h d) -> ... h d', d=self.head_k_dim)
        v = rearrange(v, '... (h d) -> ... h d', d=self.head_v_dim)
        p = rearrange(p, 'b l h -> b 1 h l')
        
        t = rearrange(t, 'b l h -> b h l')
        self.update_t(t)

        if self.feature_map is not None:
            q, k = map(lambda x: self.feature_map(x), (q, k))
        v = F.silu(v)

        # Extract gist Qs (last M tokens)
        if self.self_proto:
            p = F.sigmoid(p)
            # quantization if eval mode:
            if not self.training:
                p = p.masked_fill(p < (p.max(dim=-1, keepdim=True).values / 2), 0)
            self.update_p(p)
            gist_qs = q
        else:
            if gist_idx is not None:
                gist_idx = gist_idx.to(q.device)
                gist_qs = q
            else:
                raise NotImplementedError("GistAttention incorrectly implemented.")

        # Calculate gist attn weights
        gist_qs = torch.einsum("bmhd->bhmd", gist_qs)  # [B, G, H, D] -> [B, H, G, D]
        k_ = torch.einsum("blhd->bhdl", k)  # [B, L, H, D] -> [B, H, D, L]
        gist_attn = torch.einsum("bhmd,bhdl->bhml", gist_qs, k_) 
        # mask the gist attns 
        # 1. if token idx == gist idx, set to +inf so that K_t[idx] is initialized with k_t
        # 2. if token idx < gist idx, set to -inf so that K_t[idx] is not updated with k_t
        # 3. only accumulate k_t into K_t[idx] if token idx > gist idx
        # BUG
        # gist_attn = gist_attn.masked_fill(torch.arange(gist_attn.shape[3], device=gist_attn.device)[None, None, None, :] == gist_idx[:, None, :, None], +float('inf'))
        # gist_attn = gist_attn.masked_fill(torch.arange(gist_attn.shape[3], device=gist_attn.device)[None, None, None, :] < gist_idx[:, None, :, None], -float('inf'))  # if set to != instead of <, it reduces to normal attention

        # NOTE: this is the generalized version
        T = gist_attn.shape[3]
        # fill diag of gist_attn with +inf so that K_t[idx] is initialized with k_t
        gist_attn = gist_attn.masked_fill(torch.eye(T, device=gist_attn.device)[None, None, :, :] == 1, +float('inf'))
        # fill lower triangle of gist_attn with -inf so that K_t[idx] is not updated with tokens prior to the prototype (option 1)
        gist_attn = gist_attn.masked_fill(torch.tril(torch.ones((T, T), device=gist_attn.device), diagonal=-1)[None, None, :, :] == 1, -float('inf'))
        # fill no diagonals with -inf so that K_t[idx] is equal to k_t (option 2)
        # gist_attn = gist_attn.masked_fill(torch.eye(gist_attn.shape[3], device=gist_attn.device)[None, None, :, :] != 1, -float('inf'))

        # Assign attn weights as forget gates f
        s = torch.einsum("bhml->blhm", gist_attn)
        s = F.sigmoid(s) / self.gate_logit_normalizer

        # add log t to gist_attn, excluding the diagonal
        s = s * t.unsqueeze(1).repeat(1, T, 1, 1).masked_fill(torch.eye(T, device=gist_attn.device)[None, :, None, :] == 1, 1)
        
        f = torch.log(1 - s).to(gist_attn.dtype)

        if self.num_kv_groups > 1:
            k, v, f, s = map(lambda x: repeat(x, '... h d -> ... (h g) d', g=self.num_kv_groups), (k, v, f, s))

        recurrent_state = last_state['recurrent_state'] if last_state is not None else None
        if mode == 'fused_recurrent':
            if self.self_proto:
                o, recurrent_state = fused_recurrent_gistsa_auto(
                    q=q,
                    k=k,
                    v=v,
                    s=s,
                    g=f,
                    p=p,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=self.scale,
                    cu_seqlens=cu_seqlens,
                )
            else:
                o, recurrent_state = fused_recurrent_gistsa(
                    q=q,
                    k=k,
                    v=v,
                    s=s,
                    g=f,
                    initial_state=recurrent_state,
                    output_final_state=use_cache,
                    scale=self.scale,
                    cu_seqlens=cu_seqlens,
                    gist_idx=gist_idx,
                )
        elif mode == 'chunk':
            o, recurrent_state = chunk_gistsa(
                q=q,
                k=k,
                v=v,
                s=s,
                g=f,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                scale=self.scale,
                cu_seqlens=cu_seqlens,
                gist_idx=gist_idx,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v) if self.use_short_conv else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        o = rearrange(o, '... h d -> ... (h d)')
        o = rms_norm_linear(F.silu(o), self.g_norm.weight, self.g_norm.bias, self.o_proj.weight, self.o_proj.bias)
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)

        return o#, q, k, v, s#, None, past_key_values

    def state_size(self, **kwargs) -> int:
        state_size = self.key_dim + self.head_v_dim
        for module in self.children():
            if isinstance(module, ShortConvolution):
                state_size += module.state_size
        return state_size

    def update_p(self, p):
        self.p = p

    def update_t(self, t):
        self.t = t

    def get_auxiliary_loss(self):
        # L1 penalty
        loss = self.p.sum()
        return loss

    def log(self, logger: WandbLogger):
        # Visualize the distribution of p_t [B, 1, H, T] -> [B, T]
        for head in range(self.p.shape[2]):
            # extract the distribution for this head
            pt = self.p[:, 0, head].cpu().detach().numpy()
            # visualize the distribution
            plt.imshow(pt)                    
            # add colorbar
            plt.colorbar()
            logger.run.log({f"p_{head}": plt})
            plt.close()
            # also save a table version to keep the raw data
            logger.run.log({
                f"raw_p_{head}": wandb.Table(data=pt, columns=list(range(pt.shape[1])))
            })

            tt = self.t[:, head, :].cpu().detach().numpy()
            plt.imshow(tt)
            plt.colorbar()
            logger.run.log({f"t_{head}": plt})
            plt.close()
            logger.run.log({
                f"raw_t_{head}": wandb.Table(data=tt, columns=list(range(tt.shape[1])))
            })