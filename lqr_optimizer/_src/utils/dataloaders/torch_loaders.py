import os
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Optional
import threading
import queue
import jax

import numpy as np
import torch
from torch.utils.data import DataLoader, dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import WikiText103
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.transforms import SentencePieceTokenizer

#################################################
## Torch to Jax wrapper
#################################################
def _torch_tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
  # Ensure CPU + contiguous, then get a zero-copy NumPy view
  if t.device.type != "cpu":
    t = t.cpu()
  t = t.contiguous()
  return t.numpy()


def _coerce_dtype(arr: np.ndarray) -> np.ndarray:
  # JAX prefers int32 for indices, float32 for activations by default
  if np.issubdtype(arr.dtype, np.integer):
    if arr.dtype != np.int32:
      return arr.astype(np.int32, copy=False)
  elif np.issubdtype(arr.dtype, np.floating):
    if arr.dtype not in (np.float32, np.bfloat16):
      return arr.astype(np.float32, copy=False)
  return arr


def _map_structure(fn: Callable[[Any], Any], obj: Any) -> Any:
  if isinstance(obj, (list, tuple)):
    return type(obj)(_map_structure(fn, x) for x in obj)
  if isinstance(obj, dict):
    return {k: _map_structure(fn, v) for k, v in obj.items()}
  return fn(obj)


def _torch_struct_to_jax(struct):
  # Convert nested torch tensors -> numpy (zero-copy) -> dtype fix -> JAX arrays
  def leaf_convert(x):
    if isinstance(x, torch.Tensor):
      arr = _torch_tensor_to_numpy(x)
      arr = _coerce_dtype(arr)
      return jax.device_put(arr)  # to default device
    return x

  return _map_structure(leaf_convert, struct)


class TorchLoaderAsJaxIterator:
  """
  Wrap a PyTorch DataLoader to yield JAX DeviceArrays with background prefetch.

  Args:
      loader: PyTorch DataLoader (pin_memory=True recommended).
      prefetch: number of batches to keep buffered ahead.
      postprocess: optional callable applied to each batch AFTER device_put.
                   Example: sharding for pmap/pjit (see helper below).
  """

  def __init__(
          self,
          loader: torch.utils.data.DataLoader,
          prefetch: int = 4,
          postprocess: Optional[Callable[[Any], Any]] = None,
  ):
    self.loader = loader
    self.prefetch = max(0, int(prefetch))
    self.postprocess = postprocess
    self._q = queue.Queue(maxsize=self.prefetch if self.prefetch > 0 else 1)
    self._stop_sentinel = object()
    self._thread = None
    self._exc: Optional[BaseException] = None

  def _producer(self):
    try:
      it = iter(self.loader)
      for batch in it:
        # Convert to JAX (on device)
        jax_batch = _torch_struct_to_jax(batch)
        if self.postprocess is not None:
          jax_batch = self.postprocess(jax_batch)
        self._q.put(jax_batch)
      # signal end
      self._q.put(self._stop_sentinel)
    except BaseException as e:
      self._exc = e
      # ensure consumer unblocks
      try:
        self._q.put(self._stop_sentinel)
      except Exception:
        pass

  def __iter__(self):
    # start a fresh producer each epoch/iteration
    self._exc = None
    self._thread = threading.Thread(target=self._producer, daemon=True)
    self._thread.start()
    return self

  def __next__(self):
    item = self._q.get()
    if item is self._stop_sentinel:
      # propagate any producer exception
      if self._exc is not None:
        raise self._exc
      raise StopIteration
    return item

  def close(self):
    # No explicit cancel path (PyTorch DataLoader lacks it); rely on epoch end.
    pass


#################################################
## Torch Loaders
#################################################
def _get_bpe_tokenizer(train_file, tokenizer_save_path):
  path_to_bpe_model = tokenizer_save_path / f"{str(train_file.stem)}.model"
  if not os.path.isfile(path_to_bpe_model):
    raise ValueError(
      """Tokenizer file is required. Call the _create_bpe_tokenizer function in this file
      using the same train_file argument and a vocab size. You will find a .model file 
      in your current working directory after the _create_bpe_tokenizer has finished running. \
      Create a folder named tokenizers in the optexp workspace and put the .model file in that folder.
      Remove the call to _create_bpe_tokenizer and resume. 

  """
    )
  return SentencePieceTokenizer(str(path_to_bpe_model))

def load_wt103(
        save_path: Path,
        tokenizers_path: Path,
        batch_size: int,
        bptt: int,
        eval_batch_size: Optional[int] = 0,
):
  train_file = save_path / "WikiText103" / "wikitext-103" / "wiki.train.tokens"
  tokenizer = _get_bpe_tokenizer(train_file, tokenizer_save_path=tokenizers_path)
  return wt103_loader(
    save_path,
    batch_size,
    bptt,
    tokenizer,
    eval_batch_size=eval_batch_size,
  )

def wt103_loader(
        save_path: Path,
        batch_size,
        bptt,
        tokenizer=None,
        eval_batch_size=0,
):
  # Get splits you need for training
  if eval_batch_size==0:
    eval_batch_size = batch_size
  train_iter, val_iter, _ = WikiText103(root=save_path.parent)

  tokenizer = tokenizer or get_tokenizer("basic_english")
  vocab_path = save_path / "WikiText103" / "wiki.vocab.pkl"
  vocab_path.parent.mkdir(parents=True, exist_ok=True)

  if os.path.isfile(vocab_path):
    with open(vocab_path, "rb") as f:
      vocab = pickle.load(f)
  else:
    # Use a fresh iterator ONLY for vocab building (don’t consume your training iterator)
    vocab_train_iter = WikiText103(root=save_path.parent, split='train')
    vocab = build_vocab_from_iterator(map(tokenizer, vocab_train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    with open(vocab_path, "wb") as f:
      pickle.dump(vocab, f)

  train_path = save_path / "WikiText103" / "train.pt"
  val_path = save_path / "WikiText103" / "val.pt"

  if os.path.isfile(train_path):
    train_data = torch.load(train_path, map_location="cpu")
  else:
    train_data = tokenize_and_numify(train_iter, tokenizer, vocab=vocab)
    torch.save(train_data, train_path)

  if os.path.isfile(val_path):
    val_data = torch.load(val_path, map_location="cpu")
  else:
    val_data = tokenize_and_numify(val_iter, tokenizer, vocab=vocab)
    torch.save(val_data, val_path)
  return prepare_mixed_size_data_loader(
    train_data, val_data, batch_size, eval_batch_size, vocab, bptt, merge=None
  )

def prepare_mixed_size_data_loader(
        train_data, val_data, train_batch_size, eval_batch_size, vocab, bptt, merge
):
  class_freqs = torch.bincount(train_data)
  train_dataset = TextData(train_data, bptt)
  val_dataset = TextData(val_data, bptt)

  def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for sample in batch:
      src_batch.append(sample[0])
      tgt_batch.append(sample[1])

    return torch.transpose(torch.vstack(src_batch), 0, 1), torch.transpose(
      torch.vstack(tgt_batch), 0, 1
    ).reshape(-1)

  train_loader = DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    persistent_workers=True
  )
  eval_val_loader = DataLoader(
    val_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True,
    persistent_workers=True
  )

  input_shape = np.array([len(vocab)])
  output_shape = input_shape

  return TorchLoaderAsJaxIterator(train_loader), TorchLoaderAsJaxIterator(eval_val_loader), input_shape, output_shape, class_freqs


def tokenize_and_numify(
        raw_text_iter: dataset.IterableDataset,
        tokenizer: Callable,
        vocab: Vocab,
        cutoff: Optional[float] = None,
):
  data = [
    torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter
  ]

  if cutoff:
    x = int(cutoff * len(data))
    data = data[0:x]

  return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

class TextData(torch.utils.data.Dataset):
  def __init__(
          self, data: torch.Tensor, bptt: int, merge: Optional[int] = None
  ) -> None:
    self.data = data
    self.merge = merge
    self.tgt_len = bptt

  def __len__(self):
    return self.data.shape[0] // self.tgt_len

  def __getitem__(self, idx: int):
    seq_len = min(self.tgt_len, len(self.data) - 1 - self.tgt_len * idx)
    data = self.data[idx * self.tgt_len: idx * self.tgt_len + seq_len]
    targets = self.data[
      idx * self.tgt_len + 1: idx * self.tgt_len + 1 + seq_len
    ].reshape(-1)
    targets = (
      torch.floor(targets / self.merge).to(torch.long) if self.merge else targets
    )
    return data, targets