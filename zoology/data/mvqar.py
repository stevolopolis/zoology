
import numpy as np
import torch

from sklearn.datasets import make_blobs

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

class MVQARConfig(DataSegmentConfig):
    name: str="multiquery_ar"
    power_a: float=0.01
    num_kv_pairs: int=8
    random_non_queries: bool=True
    include_slices: bool=True
    n_dims: int=2
    n_clusters: int=3
    add_gists: bool=False

    def build(self, seed: int) -> DataSegment:
        return mvq_ar(**self.model_dump(), seed=seed)

def mvq_ar(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    power_a: float=0.01,
    num_kv_pairs: int=8,
    random_non_queries: bool=True,
    include_slices: bool=True,
    n_dims: int=2,
    n_clusters: int=3,
    add_gists: bool=False,
    **kwargs
) -> DataSegment:
    """
    Generates a variant of the synthetic data for the multi-query associative recall task as described in
    Arora,Eyuboglu, et al. "Zoology: Measuring and improving recall in efficient language models.".

    In this variant, we assign euclidean priors to tokens in the vocabulary. This means that each token in
    the first half of the vocabulary behaves equivalently to the integer vector corresponding to its index.
    Each query will then be a sequence of n_dims tokens, analogous to a 3-dimensional vector.

    For example, a query in the form of 125 277 99 will be equivalent to the vector [125, 277, 99].

    We assign labels to the queries based on some distance-driven generator function such as normal distribution.

    Example: 
        `multiquery_ar(vocab_size=12, num_kv_pairs=2, input_seq_len=16, random_non_queries=False)` 
        will generate input and label sequences of the form: 
                
                Key   Val  Key  Val            Query                         Query
        Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    2    0    0 
        Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

        The -100 labels are ignored by the loss function and metrics.
    
    We include one important note on the power law distribution. In real language data, 
    the gap between repeated bigrams follows a power law. Intuitively, if the bigram
    "common buzzard" appears in text, the probability of the bigram appearing again 
    drops the further away from the orginal mention we are. In our synthetic, we can 
    control this with the power law parameters `train_power_a` and `test_power_a`. 
    Setting these to 1.0 will result in a uniform distribution. You can visualize the
    distribution with the following code:
    ```
    space = 100
    power_a = 0.01  
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()
    plt.plot(p)
    ```

    Args:
        vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
            paper, large vocabulary sizes (>1k) can be important for highlighting 
            differences between model architectures. Defaults to 8_192.
        num_train_examples (int): The number of training examples to generate. Defaults 
            to 100_000.
        num_test_examples (int): The number of test examples to generate. Defaults to 
            3_000.
        input_seq_len (int): The length of the input sequence. Defaults to 64. In 
            In Figure 2 of the Zoology paper, we vary the input sequence length from 
            64 to 512 and the number of key-value pairs from 4 to 64.
        seed (int): The seed for the random number generator.
        num_kv_pairs (int): The number of key-value pairs.
        train_power_a (float, optional): The power for the power law distribution for 
            training data. Defaults to 0.01.
        test_power_a (float, optional): The power for the power law distribution for 
            test data. Defaults to 0.01.
        random_non_queries (bool, optional): If True, replace all the 0's (as in the 
            example above) with random values in the input. Defaults to True.

    Returns:
        SyntheticData: A SyntheticData object containing the generated train and test 
            inputs and labels.

    Raises:
        Warning: If potential data leakage is detected between the train and test sets.
    """
    assert (input_seq_len - (n_clusters if add_gists else 0)) % (n_dims + 1) == 0, f"input_seq_len - (n_clusters if add_gists else 0) must be divisible by (n_dims + 1)"
    assert vocab_size > input_seq_len, f"vocab_size must be greater than input_seq_len"
    assert num_kv_pairs * (n_dims + 1) * 2 + (n_clusters if add_gists else 0) <= input_seq_len, f"num_kv_pairs * (n_dims + 1) * 2 + (n_clusters if add_gists else 0) must be less than or equal to input_seq_len"
    # assert num_kv_pairs % n_clusters == 0, f"num_kv_pairs must be divisible by n_clusters"

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * (n_dims + 1)

    # create keys so that each key is present exactly once in each example
    # key_vocab_size = vocab_size // 2
    # key_choices = np.arange(1, key_vocab_size)
    # value_choices = np.arange(key_vocab_size, vocab_size)

    # keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    # keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

    # values_unshuffled = np.tile(value_choices, (num_examples, 1))
    # values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

    # Sample cluster centroids
    key_vocab_size = vocab_size // 2
    centroids = np.random.randint(0, key_vocab_size, (num_examples, n_clusters, n_dims))

    # Sample cluster covariances (TODO: add support for different covariances)
    cluster_std = np.ones((num_examples, n_clusters))

    # Helper function to sample vectors for a given example
    def sample_vectors_x(example_idx, is_key: bool):
        return make_blobs(
            n_samples=num_kv_pairs, 
            n_features=n_dims, 
            # centers=centroids[example_idx[0]], 
            # cluster_std=cluster_std[example_idx[0]],
            centers=n_clusters,
            shuffle=True,
            random_state=(seed * num_kv_pairs + example_idx[0]) % 4294967295,
        )[0 if is_key else 1]

    # Sample equal number of vectors per cluster using a normal distribution
    # and assign the vectors to keys and labels to values
    examples_idx = np.arange(num_examples)[:, np.newaxis]
    keys = np.apply_along_axis(sample_vectors_x, 1, examples_idx, is_key=True)
    labels = np.apply_along_axis(sample_vectors_x, 1, examples_idx, is_key=False)

    # Vectors generated by make_blobs have centroids bounded within [-10.0, 10.0]
    # We want the vectors to be bounded within [0, key_vocab_size]
    keys = keys - np.min(keys)
    keys = keys / np.max(keys)  # Normalize to [0, 1]
    keys = keys * key_vocab_size  # Scale to [0, key_vocab_size]
    keys = keys.astype(int)  # Convert to integers

    # Map the labels to random value tokens
    def map_labels_to_values(example_idx, values, labels):
        return np.take(values[example_idx[0]], labels[example_idx[0]])

    value_choices = np.arange(key_vocab_size, vocab_size)
    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)
    values = np.apply_along_axis(map_labels_to_values, 1, examples_idx, values, labels)
    
    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    for i in range(n_dims):
        kvs[:, i::(1+n_dims)] = keys[:, :, i]
    kvs[:, n_dims::(1+n_dims)] = values
    
    # gist tokens
    gists = np.apply_along_axis(np.unique, 1, values)

    # compute power law
    space = (input_seq_len - context_size - (n_clusters if add_gists else 0)) // (1+n_dims)
    p = power_a * np.arange(1, space + 1) ** (power_a-1)
    p = p / p.sum()

    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)

    # queries and answers
    queries = np.zeros((num_examples, input_seq_len - context_size - (n_clusters if add_gists else 0) + 1), dtype=np.int64)
    for i in range(n_dims):
        np.put_along_axis(queries, (gaps * (1+n_dims)) + i, values=keys[:, :, i], axis=1)

    if add_gists:
        examples = np.concatenate([
            kvs, 
            gists,
            queries
        ], axis=1)
    else:
        examples = np.concatenate([
            kvs, 
            queries
        ], axis=1)

    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, (gaps * (1+n_dims)) + context_size + n_dims + (n_clusters if add_gists else 0), values=values, axis=1)

    inputs, labels = torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])
    
    # replace all the 0 with random values
    if random_non_queries:
        inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    return DataSegment(
        inputs, 
        labels, 
        slices={"num_kv_pairs": num_kv_pairs, "input_seq_len": input_seq_len, "n_dims": n_dims, "n_clusters": n_clusters},
        kwargs={"gist_start": context_size, "n_gists": n_clusters} if add_gists else None
    )





# def associative_recall(
#     vocab_size: int=8_192,
#     num_train_examples: int=1_000,
#     num_test_examples: int=3_000,
#     input_seq_len: int=64,
#     random_non_queries: bool=True,
#     num_kv_pairs: int = 4,
#     num_queries: int = 3,
#     seed: int = 0,
# ):
#     """
#     Flexible function that generates synthetic data for both single and multi-query 
#     associative recall.

#     Example: 
#         `associative_recall(vocab_size=12, num_kv_pairs=2, num_queries=1, input_seq_len=16, random_non_queries=False)` 
#         will generate input and label sequences of the form: 
                
#                 Key   Val  Key  Val            Query                         
#         Inputs: 2     8    4    7    0    0    4    0    0    0    0    0    0    0    0 
#         Labels: -100 -100 -100 -100 -100 -100  7    -100 -100 -100 -100 -100 8    -100 -100

#         The -100 labels are ignored by the loss function and metrics.

#     Args:
#         vocab_size (int): The size of the vocabulary. As discussed in the Zoology 
#             paper, large vocabulary sizes (>1k) can be important for highlighting 
#             differences between model architectures. Defaults to 8_192.
#         num_train_examples (int): The number of training examples to generate. Defaults 
#             to 100_000.
#         num_test_examples (int): The number of test examples to generate. Defaults to 
#             3_000.
#         input_seq_len (int): The length of the input sequence. Defaults to 64. In 
#             In Figure 2 of the Zoology paper, we vary the input sequence length from 
#             64 to 512 and the number of key-value pairs from 4 to 64.
#         seed (int): The seed for the random number generator.
#         num_kv_pairs (int): The number of key-value pairs.
#         num_queries (int): The number of queries to insert into the sequence.
#         random_non_queries (bool, optional): If True, replace all the 0's (as in the 
#             example above) with random values in the input. Defaults to True.

#     Returns:
#         SyntheticData: A SyntheticData object containing the generated train and test 
#             inputs and labels.

#     Raises:
#         Warning: If potential data leakage is detected between the train and test sets.
#     """

#     train = _ar(
#         vocab_size=vocab_size,
#         num_examples=num_train_examples,
#         input_seq_len=input_seq_len,
#         seed=seed,
#         num_kv_pairs=num_kv_pairs,
#         num_queries=num_queries,
#         random_non_queries=random_non_queries
#     )
#     test = _ar(
#         vocab_size=vocab_size,
#         num_examples=num_test_examples,
#         input_seq_len=input_seq_len,
#         seed=seed + 10,  # different seed for test set
#         num_kv_pairs=num_kv_pairs,
#         num_queries=num_queries,
#         random_non_queries=random_non_queries
#     )

#     data = SyntheticData(
#         train_inputs=train_inputs,
#         train_labels=train_labels,
#         test_inputs=test_inputs,
#         test_labels=test_labels,
#     )

#     # check for data leakage:
#     train_set = set([" ".join(map(str, x)) for x in data.train_inputs.tolist()])
#     test_set = set([" ".join(map(str, x)) for x in data.test_inputs.tolist()])
#     frac_test_in_train = 1 - (len(test_set - train_set) / len(test_set))
#     if frac_test_in_train > 0.001:
#         print(
#             "WARNING: Potential data leakage detected. " 
#             f"{frac_test_in_train: 0.2f} of test examples are in the train set."
#         )
#     return data

    
# def _ar(
#     vocab_size: int,
#     num_examples: int,
#     input_seq_len: int,
#     random_non_queries: bool,
#     num_kv_pairs: int,
#     num_queries: int,
#     seed: int,
# ):
#     # SE: Using only numpy operations and no Python loops makes this alot faster. 
#     # apologies about the readbillity.
#     assert input_seq_len % 2 == 0, "input_seq_len must be even"
#     assert vocab_size > input_seq_len
#     assert num_kv_pairs * 2 + num_queries <= input_seq_len

#     np.random.seed(seed)
    
#     context_size = num_kv_pairs * 2

#     # create keys so that each key is present exactly once in each example
#     key_vocab_size = vocab_size // 2
#     key_choices = np.arange(1, key_vocab_size)
#     value_choices = np.arange(key_vocab_size, vocab_size)

#     keys_unshuffled = np.tile(key_choices, (num_examples, 1))
#     keys = np.apply_along_axis(np.random.choice, 1, keys_unshuffled, replace=False, size=num_kv_pairs)

#     values_unshuffled = np.tile(value_choices, (num_examples, 1))
#     values = np.apply_along_axis(np.random.choice, 1, values_unshuffled, replace=False, size=num_kv_pairs)

#     # create sequences
#     kvs = np.zeros((num_examples, context_size), dtype=np.int64)
#     kvs[:, 0::2] = keys
#     kvs[:, 1::2] = values

#     # create empty inputs and targets
#     # targets are filled with -100, which is ignored by the loss function and metrics
#     inputs = np.zeros((num_examples, input_seq_len), dtype=np.int64)
#     targets = np.full((num_examples, input_seq_len), dtype=np.int64, fill_value=-100)

#     # fill the first context_size tokens with the key-value pairs
#     inputs[:, 0:context_size] = kvs

#     # create a matrix of indices, which is needed to index correctly below 
#     rows = np.tile(np.arange(num_examples), (num_queries, 1)).T  

#     # sample random kv pairs to use for the queries
#     kv_idx_choices = np.arange(0, num_kv_pairs)
#     kv_idxs = np.tile(kv_idx_choices, (num_examples, 1))
#     kv_idxs = np.apply_along_axis(np.random.choice, 1, kv_idxs, replace=False, size=num_queries)
#     queries = keys[rows, kv_idxs]
#     labels = values[rows, kv_idxs]

#     # sample random positions in the last input_seq_len - context_size tokens where
#     # the queries will be inserted
#     query_pos_choices = np.arange(context_size, input_seq_len)
#     query_pos_choices = np.tile(query_pos_choices, (num_examples, 1))
#     query_pos = np.apply_along_axis(np.random.choice, 1, query_pos_choices, replace=False, size=num_queries)

#     inputs[rows, query_pos] = queries
#     targets[rows, query_pos] = labels

#     inputs, targets = torch.tensor(inputs[:, :-1]), torch.tensor(targets[:, 1:])
    
#     # replace all the 0 with random values
#     if random_non_queries:
#         inputs[inputs == 0] = torch.randint(vocab_size, size=inputs.shape)[inputs == 0]
    
#     return inputs, targets
