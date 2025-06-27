import jax
import jax.numpy as jnp
from dataclasses import dataclass
from itertools import permutations
from typing import Callable, Optional, Sequence, Tuple, Dict, Any, Union

# ================================
# Group Task Dataset Construction
# ================================

def get_group_elements_and_output_fn(dataset: str, p: int, k: int):
    """Get the group elements and output function for arithmetic or permutation tasks."""
    if dataset == "mod_sum_dataset":
        group_elements1, group_elements2 = set(range(p)), set(range(p))
        def fetch_output(a, b): return (a + b) % p
    elif dataset == "mod_subtract_dataset":
        group_elements1, group_elements2 = set(range(p)), set(range(p))
        def fetch_output(a, b): return (a - b) % p
    elif dataset == "mod_division_dataset":
        group_elements1, group_elements2 = set(range(p)), set(range(1, p))
        def fetch_output(a, b): return (a * jnp.power(b, p-2) % p) % p
    elif dataset == "permutation_group_dataset":
        perms = set(map(tuple, permutations(list(range(k)))))
        group_elements1, group_elements2 = perms, perms
        def fetch_output(a, b): return tuple(a[b[i]] for i in range(len(b)))
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    return fetch_output, group_elements1, group_elements2


def create_dataset_arrays(
    dataset: str, frac_train: float, split: str, p: int, k: int, split_seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Generate the (x, y) arrays for a given dataset split."""
    fetch_output, group_elements1, group_elements2 = get_group_elements_and_output_fn(dataset, p, k)
    all_elements = list(group_elements1.union(group_elements2))
    idx2vocab = ['o', '='] + all_elements
    vocab2idx = {vocab: idx for idx, vocab in enumerate(idx2vocab)}
    vocab_size = len(idx2vocab)

    # Enumerate all input pairs
    group1 = jnp.array(list(group_elements1))
    group2 = jnp.array(list(group_elements2))
    all_idx = jax.random.permutation(jax.random.PRNGKey(split_seed), len(group1)*len(group2))

    # Split into train/test
    split_point = int(len(all_idx) * frac_train)
    if split == 'train':
        pairs = all_idx[:split_point]
    elif split == 'test':
        pairs = all_idx[split_point:]

    # Construct dataset and store in memory:
    pairs_a = jax.vmap(lambda idx: group1[idx // len(group_elements2)])(pairs)
    pairs_b = jax.vmap(lambda idx: group2[idx % len(group_elements2)])(pairs)
    pairs_c = jax.vmap(fetch_output)(pairs_a, pairs_b)
    pairs = jnp.stack([pairs_a, pairs_b, pairs_c], axis=-1)

    def apply_mapping(x):
        a_i, b_i, c_i = x
        a_i, b_i, c_i = int(a_i), int(b_i), int(c_i)
        mapped_indices = jnp.array([vocab2idx.get(a_i), vocab2idx['o'], vocab2idx.get(b_i), vocab2idx['=']]), jnp.array(
            [vocab2idx.get(c_i) - 2, ])
        return mapped_indices

    inputs_list = []
    targets_list = []

    for el in pairs:
        inp, tgt = apply_mapping(el)
        inputs_list.append(inp)
        targets_list.append(tgt)

    # Efficiently stack once at the end
    input_array = jnp.stack(inputs_list, axis=0)
    target_array = jnp.stack(targets_list, axis=0)

    return input_array, jnp.squeeze(target_array), vocab_size

@dataclass
class AbstractDataset:
    dataset: str
    frac_train: float
    p: int
    k: int
    train_cardinality: int = 0
    test_cardinality: int = 0
    vocab_size: int = 0

    def build_dataset(self, split: str = 'train', split_seed: int = 0):
        X, Y, vocab_size = create_dataset_arrays(
            self.dataset, self.frac_train, split, self.p, self.k, split_seed=split_seed
        )
        if split == 'train':
            self.train_cardinality = X.shape[0]
        else:
            self.test_cardinality = X.shape[0]
        self.vocab_size = vocab_size
        return X, Y

class ModSumDataset(AbstractDataset):
    def __init__(self, frac_train, p, k): super().__init__("mod_sum_dataset", frac_train, p, k)
class ModSubtractDataset(AbstractDataset):
    def __init__(self, frac_train, p, k): super().__init__("mod_subtract_dataset", frac_train, p, k)
class ModDivisonDataset(AbstractDataset):
    def __init__(self, frac_train, p, k): super().__init__("mod_division_dataset", frac_train, p, k)
class PermutationGroup(AbstractDataset):
    def __init__(self, frac_train, p, k): super().__init__("permutation_group_dataset", frac_train, p, k)

class BatchingIterator:
    """An iterator yielding random batches (x, y) from precomputed arrays."""
    def __init__(self, X: jnp.ndarray, Y: jnp.ndarray, batch_size: int):
        self.X = X
        self.Y = Y
        self.n = X.shape[0]
        self.batch_size = batch_size
        self.key = jax.random.PRNGKey(42)
    def __iter__(self): return self
    def __next__(self):
        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.choice(subkey, self.n, shape=(self.batch_size,), replace=False)
        return self.X[idx], self.Y[idx]

# ==============================================
# tfds-like loader for grokking-style datasets
# ==============================================

def load_grok_ds(
    dataset: AbstractDataset,
    split: str,
    batch_size: int,
    *,
    with_info: bool = False,
    **kwargs
) -> Union[Tuple[BatchingIterator, Dict], BatchingIterator]:
    """
    Returns a generator (and optionally info dict) mimicking tfds.load:
      ds, info = tfds.load(..., as_supervised=True, with_info=True)
    """
    X, Y = dataset.build_dataset(split=split)
    ds_size = X.shape[0]

    iterator = BatchingIterator(X, Y, batch_size)
    info = {
        "vocab_size": dataset.vocab_size,
        "num_classes": len(jnp.unique(Y)),
        "ds_size": ds_size,
        "train_cardinality": getattr(dataset, "train_cardinality", None),
        "test_cardinality": getattr(dataset, "test_cardinality", None),
    }

    # To match tfds, you can request info as a second return
    if with_info:
        return iterator, info
    else:
        return iterator
