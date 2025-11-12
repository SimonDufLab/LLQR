# import os
# import pickle
# import re
# from pathlib import Path
# from typing import Any, Callable, Optional
# import threading
# import queue
# import jax
#
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, dataset
# from torchtext.data.utils import get_tokenizer
# from torchtext.datasets import WikiText103
# from torchtext.vocab import Vocab, build_vocab_from_iterator
# from torchtext.transforms import SentencePieceTokenizer
#
# #################################################
# ## Generic torch prepare dataset function
# #################################################
# def prepare_torch_dataset(
#         name: str,
# ):
#   if name=='wikitext-103':
#         return load_wt103
#   else:
#         raise ValueError(f"{name} dataset is not supported")
#
#
# #################################################
# ## Torch to Jax wrapper
# #################################################
# def _torch_tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
#   # Ensure CPU + contiguous, then get a zero-copy NumPy view
#   if t.device.type != "cpu":
#     t = t.cpu()
#   t = t.contiguous()
#   return t.numpy()
#
#
# def _coerce_dtype(arr: np.ndarray) -> np.ndarray:
#   # JAX prefers int32 for indices, float32 for activations by default
#   if np.issubdtype(arr.dtype, np.integer):
#     if arr.dtype != np.int32:
#       return arr.astype(np.int32, copy=False)
#   elif np.issubdtype(arr.dtype, np.floating):
#     if arr.dtype not in (np.float32, np.bfloat16):
#       return arr.astype(np.float32, copy=False)
#   return arr
#
#
# def _map_structure(fn: Callable[[Any], Any], obj: Any) -> Any:
#   if isinstance(obj, (list, tuple)):
#     return type(obj)(_map_structure(fn, x) for x in obj)
#   if isinstance(obj, dict):
#     return {k: _map_structure(fn, v) for k, v in obj.items()}
#   return fn(obj)
#
#
# def _torch_struct_to_jax(struct):
#   # Convert nested torch tensors -> numpy (zero-copy) -> dtype fix -> JAX arrays
#   def leaf_convert(x):
#     if isinstance(x, torch.Tensor):
#       arr = _torch_tensor_to_numpy(x)
#       arr = _coerce_dtype(arr)
#       return jax.device_put(arr)  # to default device
#     return x
#
#   return _map_structure(leaf_convert, struct)
#
#
# class TorchLoaderAsJaxIterator:
#   """
#   Wrap a PyTorch DataLoader to yield JAX DeviceArrays with background prefetch.
#
#   Args:
#       loader: PyTorch DataLoader (pin_memory=True recommended).
#       prefetch: number of batches to keep buffered ahead.
#       postprocess: optional callable applied to each batch AFTER device_put.
#                    Example: sharding for pmap/pjit (see helper below).
#   """
#
#   def __init__(
#           self,
#           loader: torch.utils.data.DataLoader,
#           prefetch: int = 4,
#           postprocess: Optional[Callable[[Any], Any]] = None,
#   ):
#     self.loader = loader
#     self.prefetch = max(0, int(prefetch))
#     self.postprocess = postprocess
#     self._q = queue.Queue(maxsize=self.prefetch if self.prefetch > 0 else 1)
#     self._stop_sentinel = object()
#     self._thread = None
#     self._exc: Optional[BaseException] = None
#
#   def _producer(self):
#     try:
#       it = iter(self.loader)
#       for batch in it:
#         # Convert to JAX (on device)
#         jax_batch = _torch_struct_to_jax(batch)
#         if self.postprocess is not None:
#           jax_batch = self.postprocess(jax_batch)
#         self._q.put(jax_batch)
#       # signal end
#       self._q.put(self._stop_sentinel)
#     except BaseException as e:
#       self._exc = e
#       # ensure consumer unblocks
#       try:
#         self._q.put(self._stop_sentinel)
#       except Exception:
#         pass
#
#   def __iter__(self):
#     # start a fresh producer each epoch/iteration
#     self._exc = None
#     self._thread = threading.Thread(target=self._producer, daemon=True)
#     self._thread.start()
#     return self
#
#   def __next__(self):
#     item = self._q.get()
#     if item is self._stop_sentinel:
#       # propagate any producer exception
#       if self._exc is not None:
#         raise self._exc
#       raise StopIteration
#     return item
#
#   def close(self):
#     # No explicit cancel path (PyTorch DataLoader lacks it); rely on epoch end.
#     pass
#
#
# #################################################
# ## Torch Loaders
# #################################################
# def _get_bpe_tokenizer(train_file, tokenizer_save_path):
#   path_to_bpe_model = tokenizer_save_path / f"{str(train_file.stem)}.model"
#   if not os.path.isfile(path_to_bpe_model):
#     raise ValueError(
#       """Tokenizer file is required. Call the _create_bpe_tokenizer function in this file
#       using the same train_file argument and a vocab size. You will find a .model file
#       in your current working directory after the _create_bpe_tokenizer has finished running. \
#       Create a folder named tokenizers in the optexp workspace and put the .model file in that folder.
#       Remove the call to _create_bpe_tokenizer and resume.
#
#   """
#     )
#   return SentencePieceTokenizer(str(path_to_bpe_model))
#
# def load_wt103(
#         save_path: Path,
#         tokenizers_path: Path,
#         batch_size: int,
#         bptt: int,
#         eval_batch_size: Optional[int] = 0,
# ):
#   train_file = save_path / "wikitext-103" / "wiki.train.tokens"
#   try:
#     tokenizer = _get_bpe_tokenizer(train_file, tokenizer_save_path=tokenizers_path)
#   except ValueError:
#     tokenizer = None
#   return wt103_loader(
#     save_path,
#     batch_size,
#     bptt,
#     tokenizer,
#     eval_batch_size=eval_batch_size,
#   )
#
# def wt103_loader(
#         save_path: Path,
#         batch_size,
#         bptt,
#         tokenizer=None,
#         eval_batch_size=0,
# ):
#   # Get splits you need for training
#   if eval_batch_size==0:
#     eval_batch_size = batch_size
#   train_iter = WikiText103(root=save_path.parent, split='train')
#
#   tokenizer = tokenizer or get_tokenizer("basic_english")
#   vocab_path = save_path / "wiki.vocab.pkl"
#   vocab_path.parent.mkdir(parents=True, exist_ok=True)
#
#   if os.path.isfile(vocab_path):
#     with open(vocab_path, "rb") as f:
#       vocab = pickle.load(f)
#   else:
#     # Use a fresh iterator ONLY for vocab building (don’t consume your training iterator)
#     vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=["<unk>"])
#     vocab.set_default_index(vocab["<unk>"])
#     with open(vocab_path, "wb") as f:
#       pickle.dump(vocab, f)
#
#   train_iter, val_iter, _ = WikiText103(root=save_path.parent)
#
#   train_path = save_path / "train.pt"
#   val_path = save_path / "val.pt"
#
#   if os.path.isfile(train_path):
#     train_data = torch.load(train_path, map_location="cpu")
#   else:
#     train_data = tokenize_and_numify(train_iter, tokenizer, vocab=vocab)
#     torch.save(train_data, train_path)
#
#   if os.path.isfile(val_path):
#     val_data = torch.load(val_path, map_location="cpu")
#   else:
#     val_data = tokenize_and_numify(val_iter, tokenizer, vocab=vocab)
#     torch.save(val_data, val_path)
#   return prepare_mixed_size_data_loader(
#     train_data, val_data, batch_size, eval_batch_size, vocab, bptt, merge=None
#   )
#
# def prepare_mixed_size_data_loader(
#         train_data, val_data, train_batch_size, eval_batch_size, vocab, bptt, merge
# ):
#   class_freqs = torch.bincount(train_data)
#   info = {"class_freqs": class_freqs}
#   train_dataset = TextData(train_data, bptt)
#   val_dataset = TextData(val_data, bptt)
#
#   def collate_fn(batch):
#     src_batch, tgt_batch = [], []
#     for sample in batch:
#       src_batch.append(sample[0])
#       tgt_batch.append(sample[1])
#
#     return torch.transpose(torch.vstack(src_batch), 0, 1), torch.transpose(
#       torch.vstack(tgt_batch), 0, 1
#     ).reshape(-1)
#
#   train_loader = DataLoader(
#     train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True,
#     persistent_workers=True
#   )
#   eval_val_loader = DataLoader(
#     val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True,
#     persistent_workers=True
#   )
#   info["num_classes"] = len(vocab)
#   info["ds_size"] = len(train_dataset)
#
#   info["input_shape"] = np.array([len(vocab)])
#   info["output_shape"] = info["input_shape"]
#
#   return TorchLoaderAsJaxIterator(train_loader), TorchLoaderAsJaxIterator(eval_val_loader), info
#
#
# def tokenize_and_numify(
#         raw_text_iter: dataset.IterableDataset,
#         tokenizer: Callable,
#         vocab: Vocab,
#         cutoff: Optional[float] = None,
# ):
#   data = [
#     torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
#   ]
#
#   if cutoff:
#     x = int(cutoff * len(data))
#     data = data[0:x]
#
#   return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
#
# class TextData(torch.utils.data.Dataset):
#   def __init__(
#           self, data: torch.Tensor, bptt: int, merge: Optional[int] = None
#   ) -> None:
#     self.data = data
#     self.merge = merge
#     self.tgt_len = bptt
#
#   def __len__(self):
#     return self.data.shape[0] // self.tgt_len
#
#   def __getitem__(self, idx: int):
#     seq_len = min(self.tgt_len, len(self.data) - 1 - self.tgt_len * idx)
#     data = self.data[idx * self.tgt_len: idx * self.tgt_len + seq_len]
#     targets = self.data[
#       idx * self.tgt_len + 1: idx * self.tgt_len + 1 + seq_len
#     ].reshape(-1)
#     targets = (
#       torch.floor(targets / self.merge).to(torch.long) if self.merge else targets
#     )
#     return data, targets

# hf_wt103_jax.py
import os
import re
import pickle
import threading
import queue
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Iterable, Iterator, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from datasets import load_dataset

#################################################
## Entry point selector (kept compatible)
#################################################
def prepare_hf_dataset(name: str):
    # Kept the same public API name to avoid touching your caller
    if name == 'wikitext-103':
        return load_wt103
    else:
        raise ValueError(f"{name} dataset is not supported")


#################################################
## JAX prefetching iterator (no Torch required)
#################################################
class LoaderAsJaxIterator:
    """
    Wrap any Python iterator that yields NumPy arrays (or nested structures)
    and device_put them with a background prefetching thread.
    """
    def __init__(
        self,
        source_iter: Iterable,
        prefetch: int = 4,
        postprocess: Optional[Callable[[Any], Any]] = None,
    ):
        self.source_iter = source_iter
        self.prefetch = max(0, int(prefetch))
        self.postprocess = postprocess
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=self.prefetch if self.prefetch > 0 else 1)
        self._stop = object()
        self._exc: Optional[BaseException] = None
        self._thread: Optional[threading.Thread] = None

    def _to_device(self, obj):
        if isinstance(obj, np.ndarray):
            return jax.device_put(obj)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    def _producer(self):
        try:
            for batch in self.source_iter:
                batch = self._to_device(batch)
                if self.postprocess is not None:
                    batch = self.postprocess(batch)
                self._q.put(batch)
            self._q.put(self._stop)
        except BaseException as e:
            self._exc = e
            try:
                self._q.put(self._stop)
            except Exception:
                pass

    def __iter__(self):
        self._exc = None
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        item = self._q.get()
        if item is self._stop:
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item


#################################################
## Tokenizers
#################################################
# Simple "basic_english"-ish tokenizer: lowercase, split on words & punctuation
_BASIC_ENGLISH_RE = re.compile(r"[A-Za-z0-9]+|\S", re.UNICODE)

def basic_english(text: str):
    text = text.lower()
    return _BASIC_ENGLISH_RE.findall(text)


#################################################
## SentencePiece (optional)
#################################################
class SentencePieceWrapper:
    def __init__(self, model_path: Path):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise RuntimeError("Please `pip install sentencepiece` to use SentencePieceTokenizer.") from e
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.sp.encode(text, out_type=int), dtype=np.int32)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())


#################################################
## Vocab (word-level) utilities
#################################################
class Vocab:
    def __init__(self, stoi: Dict[str, int], unk_token: str = "<unk>"):
        self.stoi = dict(stoi)
        self.itos = [None] * len(stoi)
        for s, i in self.stoi.items():
            self.itos[i] = s
        self.unk_idx = self.stoi.get(unk_token, 0)

    def __len__(self):
        return len(self.itos)

    def encode_tokens(self, toks: Iterable[str]) -> np.ndarray:
        # int32 for JAX embedding indices
        return np.asarray([self.stoi.get(t, self.unk_idx) for t in toks], dtype=np.int32)


def build_or_load_vocab(save_path: Path, train_texts: Iterable[str], tokenizer_fn: Callable[[str], Iterable[str]]):
    vocab_path = save_path / "wiki.vocab.pkl"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    if vocab_path.exists():
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    # Build vocab from training iterator
    counter: Dict[str, int] = {}
    for line in train_texts:
        for tok in tokenizer_fn(line):
            counter[tok] = counter.get(tok, 0) + 1

    # Reserve <unk> at index 0
    sorted_items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    stoi = {"<unk>": 0}
    for tok, _ in sorted_items:
        if tok not in stoi:
            stoi[tok] = len(stoi)

    vocab = Vocab(stoi)
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    return vocab


#################################################
## Dataset → token arrays
#################################################
def _encode_split_to_ids(
    texts: Iterable[str],
    *,
    tokenizer_fn: Optional[Callable[[str], Iterable[str]]] = None,
    vocab: Optional[Vocab] = None,
    sp: Optional[SentencePieceWrapper] = None,
) -> np.ndarray:
    ids: list[np.ndarray] = []
    if sp is not None:
        for line in texts:
            arr = sp.encode(line)
            if arr.size > 0:
                ids.append(arr)
    else:
        assert tokenizer_fn is not None and vocab is not None
        for line in texts:
            toks = tokenizer_fn(line)
            if not toks:
                continue
            arr = vocab.encode_tokens(toks)
            if arr.size > 0:
                ids.append(arr)
    if not ids:
        return np.empty((0,), dtype=np.int32)
    return np.concatenate(ids, dtype=np.int32)


#################################################
## NumPy "TextData"
#################################################
class TextData:
    """
    Given a 1D array of token IDs, emit non-overlapping windows of length `bptt`
    as (data, targets) where targets are the next-token shift of data.
    Length = len(data)//bptt (like your Torch version).
    """
    def __init__(self, data: np.ndarray, bptt: int, merge: Optional[int] = None):
        assert data.ndim == 1
        self.data = data
        self.tgt_len = int(bptt)
        self.merge = merge

    def __len__(self):
        return int(self.data.shape[0] // self.tgt_len)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        start = idx * self.tgt_len
        end = start + self.tgt_len
        x = self.data[start:end]
        y = self.data[start + 1 : end + 1]
        # clip the last item if needed (match your min logic)
        if y.shape[0] < x.shape[0]:
            x = x[: y.shape[0]]
        if self.merge:
            y = (y // self.merge).astype(np.int32, copy=False)
        return x.astype(np.int32, copy=False), y.astype(np.int32, copy=False)


def _batch_iterator(
    dataset: TextData,
    batch_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yields batches shaped like your previous collate_fn:
      - stack B sequences [B, T]
      - then transpose to [T, B]
      - targets flattened to [T*B]
    (No shuffle to match your previous code.)
    """
    n = len(dataset)
    T = dataset.tgt_len
    i = 0
    while i < n:
        j = min(i + batch_size, n)
        xs = []
        ys = []
        for k in range(i, j):
            x, y = dataset[k]           # [T], [T]
            xs.append(x)
            ys.append(y)
        X = np.stack(xs, axis=0)        # [B, T]
        Y = np.stack(ys, axis=0)        # [B, T]
        X = X.transpose(1, 0).copy()    # [T, B]
        Y = Y.transpose(1, 0).reshape(-1).copy()  # [T*B]
        yield X, Y
        i = j


#################################################
## Public API: load_wt103 / wt103_loader
#################################################
def load_wt103(
    save_path: Path,
    tokenizers_path: Path,
    batch_size: int,
    bptt: int,
    eval_batch_size: Optional[int] = 0,
):
    """
    Entry point kept for compatibility with your caller.
    Tries SentencePiece first if a model is found at tokenizers_path/<something>.model,
    otherwise falls back to basic_english word-level vocab.
    """
    # Try to find a .model in tokenizers_path; if present, use SP
    sp_model = None
    if tokenizers_path and tokenizers_path.exists():
        for p in tokenizers_path.glob("*.model"):
            sp_model = p
            break

    if sp_model is not None:
        tokenizer = SentencePieceWrapper(sp_model)
        vocab = None
    else:
        tokenizer = None
        vocab = None  # built below for basic_english

    return wt103_loader(
        save_path=save_path,
        batch_size=batch_size,
        bptt=bptt,
        tokenizer=tokenizer,
        eval_batch_size=eval_batch_size or batch_size,
        vocab=vocab,
    )


def wt103_loader(
    save_path: Path,
    batch_size: int,
    bptt: int,
    tokenizer: Optional[SentencePieceWrapper] = None,
    eval_batch_size: int = 0,
    vocab: Optional[Vocab] = None,
):
    """
    HF-only pipeline. Caches token ids to save_path/{train.npy,val.npy}.
    Returns JAX-prefetching iterators and an info dict.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    train_ids_path = save_path / "train.npy"
    val_ids_path   = save_path / "val.npy"
    vocab_path     = save_path / "wiki.vocab.pkl"

    # 1) Load HF dataset (raw text)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = (x["text"] for x in ds["train"])
    val_texts   = (x["text"] for x in ds["validation"])

    # 2) Tokenize → ids (cached)
    if tokenizer is not None:
        # SentencePiece mode
        if not train_ids_path.exists():
            train_ids = _encode_split_to_ids(train_texts, sp=tokenizer)
            np.save(train_ids_path, train_ids)
        else:
            train_ids = np.load(train_ids_path, mmap_mode="r")
        if not val_ids_path.exists():
            val_ids = _encode_split_to_ids(val_texts, sp=tokenizer)
            np.save(val_ids_path, val_ids)
        else:
            val_ids = np.load(val_ids_path, mmap_mode="r")
        num_classes = tokenizer.vocab_size
        vocab_obj = None
    else:
        # Basic English mode with our own vocab
        if vocab is None:
            # Build or load vocab from training split
            vocab_obj = build_or_load_vocab(save_path, (x["text"] for x in ds["train"]), basic_english)
        else:
            vocab_obj = vocab

        if not train_ids_path.exists():
            train_ids = _encode_split_to_ids((x["text"] for x in ds["train"]), tokenizer_fn=basic_english, vocab=vocab_obj)
            np.save(train_ids_path, train_ids)
        else:
            train_ids = np.load(train_ids_path, mmap_mode="r")

        if not val_ids_path.exists():
            val_ids = _encode_split_to_ids((x["text"] for x in ds["validation"]), tokenizer_fn=basic_english, vocab=vocab_obj)
            np.save(val_ids_path, val_ids)
        else:
            val_ids = np.load(val_ids_path, mmap_mode="r")

        num_classes = len(vocab_obj)

    # Ensure arrays are int32 for JAX embedding indices
    train_ids = np.asarray(train_ids, dtype=np.int32)
    val_ids   = np.asarray(val_ids,   dtype=np.int32)

    # 3) Build datasets
    train_dataset = TextData(train_ids, bptt=bptt)
    val_dataset   = TextData(val_ids,   bptt=bptt)

    # 4) Build iterators (no torch DataLoader; we keep shapes identical)
    def make_iter(ds_obj: TextData, batch_size: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        return _batch_iterator(ds_obj, batch_size)

    train_iter_np = make_iter(train_dataset, batch_size)
    val_iter_np   = make_iter(val_dataset,   eval_batch_size or batch_size)

    train_loader = LoaderAsJaxIterator(train_iter_np, prefetch=4)
    eval_val_loader = LoaderAsJaxIterator(val_iter_np, prefetch=2)

    # 5) Stats/info
    # class frequencies over the training ids
    # (np.bincount length must cover all classes)
    class_freqs = np.bincount(train_ids, minlength=num_classes).astype(np.int64)
    info = {
        "class_freqs": class_freqs,
        "num_classes": int(num_classes),
        "ds_size": int(len(train_dataset)),
        "input_shape": np.array([num_classes]),
        "output_shape": np.array([num_classes]),
    }

    return train_loader, eval_val_loader, info
