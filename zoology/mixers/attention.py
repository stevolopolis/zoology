import torch 
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import matplotlib.pyplot as plt
import time


class SelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0, gist=False):
        super().__init__()
        self.dropout_p = attention_dropout
        self.gist = gist

    def forward(self, qkv, gist_start=None, n_gists=None, query_start=None, gist_attention_mask=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
            causal: if passed, will override self.causal
        """
        seqlen = qkv.shape[1]
        q, k, v = qkv.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        # mask all the tokens prior to gist start from tokens after gist_start + n_gists
        if self.gist:
            if gist_attention_mask is not None and query_start is not None:
                gist_mask = gist_attention_mask
                gist_mask = gist_mask[:, None, :].repeat(1, causal_mask.shape[1], 1)
                gist_mask[:, :, query_start:] = True
                gist_mask[:, :query_start] = True
                causal_mask = causal_mask[None, ...].repeat(q.shape[0], 1, 1)
                causal_mask[~gist_mask] = -10000.0
                causal_mask = causal_mask[:, None, ...]  # expand the head dimension
            elif gist_start is not None and n_gists is not None:
                causal_mask[gist_start+n_gists:, :gist_start] = -10000.0
            else:
                raise NotImplementedError("GistAttention incorrectly implemented.")

        print(scores.shape, causal_mask.shape)
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention_drop = F.dropout(attention, self.dropout_p if self.training else 0.0)
        output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
        return output


class MHA(nn.Module):
    """Multi-head self-attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        layer_idx: int=None,
        gist: bool=False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_model % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.Wqkv = nn.Linear(
            d_model, 3 * d_model, bias=bias
        )
        self.inner_attn = SelfAttention(attention_dropout=dropout, gist=gist)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, **kwargs):
        """"""
        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        context = self.inner_attn(qkv, **kwargs)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * sequence_length