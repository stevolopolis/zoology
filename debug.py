from zoology.data.mqgar import MQGARConfig
from zoology.data.mvqar import MVQARConfig
from zoology.data.associative_recall import MQARConfig
from zoology.data.utils import DataSegment

import numpy as np
import matplotlib.pyplot as plt


n_dims = 2
n_clusters = 4
num_kv_pairs = 1024
add_gists = True

config = MVQARConfig(
    vocab_size=16_192,
    input_seq_len=(n_dims + 1) * num_kv_pairs * 2 + (n_clusters if add_gists else 0),
    num_examples=5,
    num_kv_pairs=num_kv_pairs,
    random_non_queries=False,
    n_dims=n_dims,
    n_clusters=n_clusters,
    query_max_len=n_dims,
    n_unique_values=n_clusters,
    # min_keys=False,
    add_gists=add_gists,
)

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