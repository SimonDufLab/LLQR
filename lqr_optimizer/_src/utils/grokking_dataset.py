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
        group_elements = set(range(p))
        def fetch_output(a, b): return (a + b) % p
    elif dataset == "mod_subtract_dataset":
        group_elements = set(range(p))
        def fetch_output(a, b): return (a - b) % p
    elif dataset == "mod_division_dataset":
        group_elements = set(range(p))
        def fetch_output(a, b): return (a * pow(b, p-2, p)) % p
    elif dataset == "permutation_group_dataset":
        perms = set(map(tuple, permutations(range(k))))
        group_elements = perms
        def fetch_output(a, b): return tuple(a[b[i]] for i in range(len(b)))
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")
    return fetch_output, group_elements, group_elements

def build_vocab(group_elements) -> Tuple[Dict[Any, int], Dict[int, Any], int]:
    """Build vocab mappings. First two are operator/equal, then group elements."""
    all_elements = list(group_elements)
    idx2vocab = ['o', '='] + all_elements
    vocab2idx = {v: i for i, v in enumerate(idx2vocab)}
    return vocab2idx, idx2vocab, len(idx2vocab)

def create_dataset_arrays(
    dataset: str, frac_train: float, split: str, p: int, k: int, split_seed: int = 0
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Generate the (x, y) arrays for a given dataset split."""
    fetch_output, group_elements1, group_elements2 = get_group_elements_and_output_fn(dataset, p, k)
    vocab2idx, idx2vocab, vocab_size = build_vocab(group_elements1.union(group_elements2))

    # Enumerate all input pairs
    group1 = list(group_elements1)
    group2 = list(group_elements2)
    pairs = [(a, b) for a in group1 for b in group2]
    all_idx = jax.random.permutation(jax.random.PRNGKey(split_seed), len(pairs))

    # Split into train/test
    split_point = int(len(pairs) * frac_train)
    if split == 'train':
        idxs = all_idx[:split_point]
    else:
        idxs = all_idx[split_point:]

    # Prepare x and y arrays
    X, Y = [], []
    for idx in idxs:
        a, b = pairs[idx]
        c = fetch_output(a, b)
        x = [vocab2idx[a], vocab2idx['o'], vocab2idx[b], vocab2idx['=']]
        y = vocab2idx[c] - 2  # Shift so targets start at 0
        X.append(x)
        Y.append(y)
    return jnp.array(X, dtype=jnp.int32), jnp.array(Y, dtype=jnp.int32), vocab_size

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
