from zoology.data.mqgar import MQGARConfig
from zoology.data.mvqar import MVQARConfig
from zoology.data.associative_recall import MQARConfig
from zoology.data.utils import DataSegment
from zoology.experiments.models_repo import add_gist_attention, add_gistsa
from zoology.data.utils import prepare_data
from zoology.config import DataConfig
from zoology.model import LanguageModel
from zoology.train import Trainer

import numpy as np
import matplotlib.pyplot as plt
import torch
from einops import rearrange



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 16_192
n_dims = 2
n_clusters = 8
num_kv_pairs = 32
add_gists = False
input_seq_len = (n_dims + 1) * num_kv_pairs * 2 + (n_clusters if add_gists else 0)
# input_seq_len = 64
random_gists = False
manual_gist_mode = "first_unique"


config = MVQARConfig(
    vocab_size=VOCAB_SIZE,
    input_seq_len=(n_dims + 1) * num_kv_pairs * 2 + (n_clusters if add_gists else 0),
    num_examples=16,
    num_kv_pairs=num_kv_pairs,
    random_non_queries=False,
    n_dims=n_dims,
    n_clusters=n_clusters,
    query_max_len=n_dims,
    n_unique_values=n_clusters,
    # min_keys=False,
    add_gists=add_gists,
    random_gists=random_gists,  
    manual_gist_mode=manual_gist_mode,
)


model_factory_kwargs = {
    "state_mixer": dict(name="torch.nn.Identity", kwargs={}),
    "vocab_size": VOCAB_SIZE,
}

# define this conv outside of if/else block because it is used in multiple models
conv_mixer = dict(
    name="zoology.mixers.base_conv.BaseConv",
    kwargs={
        "l_max": input_seq_len,
        "kernel_size": 3,
        "implicit_long_conv": True,
    }
)


# Test data
def test_data():
    data = DataSegment.from_config(config)

    print(data.inputs.shape)
    print(data.inputs[0])
    print(data.labels[0])
    print(len(np.unique(data.labels[0])))

    # Visualize the data
    n_plots = 4
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(n_plots):
        label_idx = np.where(data.labels[i] != -100)[0]
        y = data.labels[i][label_idx]
        x = np.stack([data.inputs[i][label_idx - n_dims + k] for k in range(1, n_dims + 1)], axis=1)

        scatter = axs[i // 2, i % 2].scatter(x[:, 0], x[:, 1] if n_dims == 2 else np.zeros_like(x[:, 0]), c=y)
        legend = axs[i // 2, i % 2].legend(*scatter.legend_elements(), title="Labels")
        axs[i // 2, i % 2].add_artist(legend)

    plt.savefig(f"debug.pdf")
    plt.close()


def test_gist_mask():
    data = DataSegment.from_config(config)

    inputs_ = data.inputs.cpu().numpy()[0]
    gist_attention_mask = data.kwargs["gist_attention_mask"].cpu().numpy()[0]
    selected_tokens = inputs_[gist_attention_mask]
    labels_ = data.labels.cpu().numpy()[0]
    labels = labels_[labels_ != -100]

    print(np.sort(selected_tokens))
    print(np.sort(np.unique(labels)))


# Test model implementation
def test_model():
    models = add_gistsa([], conv_mixer, input_seq_len, model_factory_kwargs, self_proto=True)
    # models = add_gist_attention([], conv_mixer, input_seq_len, model_factory_kwargs)
    model = LanguageModel(config=models[0])
    model.eval()
    model.to(DEVICE)

    data = DataConfig(
        train_configs=[config],
        test_configs=[config],
        # can pass a tuple if you want a different batch size for train and test
        batch_size=4,
        cache_dir="/scratch/sl12886/tts_icl/zoology/data",
        force_cache=True
    )

    _, test_dataloader = prepare_data(data)

    inputs, targets, slices, kwargs = next(iter(test_dataloader))
    inputs2 = inputs.clone()
    targets2 = targets.clone()

    kwargs1 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    model.mode = "fused_recurrent"
    logits1 = model(inputs, **kwargs1)
    logits1.sum().backward()
    grad1 = next(model.parameters()).grad

    inputs2, targets2 = inputs2.to(DEVICE), targets2.to(DEVICE)
    model.mode = "chunk"
    logits2 = model(inputs2, **kwargs1)
    logits2.sum().backward()
    grad2 = next(model.parameters()).grad

    print(kwargs1["gist_idx"])

    print(torch.allclose(logits1, logits2))
    print(torch.abs(logits1 - logits2).mean())

    print(torch.allclose(grad1, grad2))
    print(torch.abs(grad1 - grad2).mean())
    print(grad1)
    print(grad2)

    for name, p in model.named_parameters():
        grad = p.grad
        print(name)
        if grad is None:
            print("No gradient")
            continue
        print(grad.shape)
        print(grad)
        grad = grad.cpu().detach().numpy()
        if len(grad.shape) == 2:
            plt.matshow(grad)
            plt.colorbar()
            plt.savefig(f"debug_{name}.pdf")
            plt.close()



def test_trainer():
    models = add_gistsa([], conv_mixer, input_seq_len, model_factory_kwargs, self_proto=True)
    # models = add_gist_attention([], conv_mixer, input_seq_len, model_factory_kwargs)
    model = LanguageModel(config=models[0])

    trainer = Trainer(model)
    trainer.save(models[0])


def test_fla_gistsa_naive():
    from fla.ops.gistsa import naive_recurrent_gistsa, chunk_gistsa
    B, H, T, D, M = 2, 2, 8, 4, 3
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    s = torch.randn(B, H, T, M)
    g = torch.randn(B, H, T, M)
    gist_idx = torch.tensor([[0, 1, 2],
                             [1, 2, 3]])

    # test fwd
    o1 = naive_recurrent_gistsa(q, k, v, s, g, gist_idx=gist_idx)
    o2 = chunk_gistsa(q, k, v, s, g, gist_idx=gist_idx)

    print(torch.allclose(o1, o2))
    print(torch.abs(o1 - o2).mean())

    # test bwd


def test_gistsa_masks():
    B, H, T, M = 2, 2, 64, 64
    gist_attn = torch.randn(B, H, M, T)
    gist_idx = torch.arange(M).expand(B, -1)

    gist_attn = gist_attn.masked_fill(torch.arange(T, device=gist_attn.device)[None, None, None, :] == gist_idx[:, None, :, None], +float('inf'))
    gist_attn = gist_attn.masked_fill(torch.arange(T, device=gist_attn.device)[None, None, None, :] < gist_idx[:, None, :, None], -float('inf'))
    gist_attn = torch.nn.functional.sigmoid(gist_attn)

    plt.matshow(gist_attn[0, 0].cpu().numpy())
    plt.colorbar()
    plt.savefig("debug.pdf")
    plt.close()

    print(gist_attn[0, 0])


def test_pseudo_attn():
    """
    For this test to pass, you need to:
    1. set the gate_logit_normalizer to 1.0 in the GistSlotAttention constructor
    2. set the scale to None in the GistSlotAttention constructor
    3. comment out the rms_norm_linear call in the GistSlotAttention forward method
    4. change the second gist_attn mask to != instead of <
    5. return q, k, v, s in the GistSlotAttention forward method
    """
    from zoology.mixers.gistsa import GistSlotAttention
    from zoology.mixers.attention import SelfAttention

    D = 256
    B, H, T, M = 2, 2, 64, 64
    hidden_states = torch.randn(B, T, D, device="cuda")
    gist_idx = torch.arange(M, device="cuda").expand(B, -1)
    gist_idx[:, ::2] = 0
    
    gsa = GistSlotAttention(d_model=D, mode="fused_recurrent", gate_logit_normalizer=1.0, scale=None).to("cuda")
    attn = SelfAttention(gist=True).to("cuda")

    o, q, k, v, s = gsa(hidden_states, gist_idx=gist_idx)
    o2 = attn(qkv=torch.stack([q, k, v], dim=2), gist_idx=gist_idx)
    o2 = rearrange(o2, "... h d -> ... (h d)")
    print(torch.allclose(o, o2))
    print(torch.abs(o - o2).mean())

    plt.matshow(s[0, :, 0].cpu().detach().numpy())
    plt.colorbar()
    plt.savefig("test.pdf")
    plt.close()



if __name__ == "__main__":
    # test_data()
    test_model()
    # test_gist_mask()
    # test_trainer()
    # test_fla_gistsa_naive()
    # test_fla_softmax_naive()
    # test_gistsa_masks()
    # test_pseudo_attn()