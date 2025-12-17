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



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 16_192
n_dims = 2
n_clusters = 3
num_kv_pairs = 16
add_gists = False
input_seq_len = (n_dims + 1) * num_kv_pairs * 2 + (n_clusters if add_gists else 0)
random_gists = False


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

    kwargs1 = {k: v for k, v in kwargs.items() if not isinstance(v, torch.Tensor)}
    kwargs2 = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}

    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
    logits1 = model(inputs, **kwargs1)
    logits2 = model(inputs, **kwargs2)

    print(torch.allclose(logits1, logits2))
    print(torch.abs(logits1 - logits2).mean())


def test_trainer():
    models = add_gistsa([], conv_mixer, input_seq_len, model_factory_kwargs, self_proto=True)
    # models = add_gist_attention([], conv_mixer, input_seq_len, model_factory_kwargs)
    model = LanguageModel(config=models[0])

    trainer = Trainer(model)
    trainer.save(models[0])


def test_fla_gistsa_naive():
    from fla.ops.gistsa.naive import naive_recurrent_gistsa
    B, H, T, D, M = 2, 2, 8, 4, 3
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    s = torch.randn(B, H, T, M)
    g = torch.randn(B, H, T, M)
    gist_idx = torch.tensor([[0, 1, 2],
                             [1, 2, 3]])
    o = naive_recurrent_gistsa(q, k, v, s, g, gist_idx=gist_idx)


def test_fla_softmax_naive():
    from fla.ops.gistsa.chunk import naive_softmax_fwd, naive_softmax_bwd, softmax_fwd, softmax_bwd
    B, H, T, D, M = 2, 2, 8, 4, 3
    qk = torch.randn(B, H, T, M)
    gist_idx = torch.tensor([[0, 1, 2],
                             [1, 2, 3]])

    p1 = naive_softmax_fwd(qk, gist_idx)
    p2 = softmax_fwd(qk, dtype=torch.float)
    print(torch.allclose(p1, p2))
    print(torch.abs(p1 - p2).mean())

    dqv = torch.randn(B, H, T, D)
    dok1 = naive_softmax_bwd(p1, dqv, gist_idx)
    dok2 = softmax_bwd(p2, dqv, dtype=torch.float)
    print(torch.allclose(dok1, dok2))


if __name__ == "__main__":
    # test_data()
    # test_model()
    # test_gist_mask()
    # test_trainer()
    # test_fla_gistsa_naive()
    test_fla_softmax_naive()