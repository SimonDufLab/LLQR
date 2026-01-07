import re
import pickle
import threading
import queue
from pathlib import Path
from typing import Any, Callable, Optional, Dict, Iterable, Iterator, Tuple

import numpy as np
import jax
from datasets import load_dataset
from transformers import GPT2TokenizerFast

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
  Wrap an epoch *factory* (callable returning an iterator of batches)
  and device_put them with a background prefetching thread.

  If repeat=True, it recreates a fresh epoch iterator when one finishes
  (for infinite train). For eval, usually repeat=False.
  """

  def __init__(
          self,
          source: Iterable | Callable[[], Iterable],
          prefetch: int = 4,
          postprocess: Optional[Callable[[Any], Any]] = None,
          repeat: bool = False,
  ):
    self.source = source
    self.prefetch = max(0, int(prefetch))
    self.postprocess = postprocess
    self.repeat = repeat
    self._init_state()  # <— important

  def _init_state(self):
    # (Re)initialize runtime state
    self._q: "queue.Queue[Any]" = queue.Queue(
      maxsize=self.prefetch if self.prefetch > 0 else 1
    )
    self._stop = object()
    self._exc: Optional[BaseException] = None
    self._thread: Optional[threading.Thread] = None
    self._started = False

  def reset(self):
    """
    Reset the iterator so it can be re-used for a new evaluation run.
    Only works reliably if `source` is a factory (callable) that
    returns a fresh epoch iterator each time.
    """
    self._init_state()

  def _to_device(self, obj):
    if isinstance(obj, np.ndarray):
      return jax.device_put(obj)
    if isinstance(obj, (list, tuple)):
      return type(obj)(self._to_device(x) for x in obj)
    if isinstance(obj, dict):
      return {k: self._to_device(v) for k, v in obj.items()}
    return obj

  def _get_iter(self) -> Iterable:
    # If we got a factory function, call it; else assume it's an iterable
    if callable(self.source):
      return self.source()
    return self.source

  def _producer_once(self):
    for batch in self._get_iter():
      batch = self._to_device(batch)
      if self.postprocess is not None:
        batch = self.postprocess(batch)
      self._q.put(batch)

  def _producer(self):
    try:
      while True:
        self._producer_once()
        if not self.repeat:
          break
      self._q.put(self._stop)
    except BaseException as e:
      self._exc = e
      try:
        self._q.put(self._stop)
      except Exception:
        pass

  def _ensure_started(self):
    if not self._started:
      self._exc = None
      self._thread = threading.Thread(target=self._producer, daemon=True)
      self._thread.start()
      self._started = True

  def __iter__(self):
    self._ensure_started()
    return self

  def __next__(self):
    self._ensure_started()
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
        vocab: Optional["Vocab"] = None,
        sp: Optional["SentencePieceWrapper"] = None,
        hf_tokenizer: Optional[Any] = None,  # e.g., GPT2TokenizerFast
        add_special_tokens: bool = False,  # usually False for LM streams
) -> np.ndarray:
  """
  Convert an iterable of text lines into a single flat int32 array of token IDs.
  Supports three modes (priority order):
    1) hf_tokenizer (e.g., GPT2TokenizerFast)
    2) sp (SentencePieceWrapper)
    3) tokenizer_fn + vocab (Basic English word-level)
  """
  ids: list[np.ndarray] = []

  if hf_tokenizer is not None:
    # HuggingFace tokenizer path (recommended for GPT-2)
    for line in texts:
      # GPT-2 often trained without special tokens for raw LM streams
      arr = hf_tokenizer.encode(line, add_special_tokens=add_special_tokens)
      if arr:
        ids.append(np.asarray(arr, dtype=np.int32))

  elif sp is not None:
    # SentencePiece path
    for line in texts:
      arr = sp.encode(line)
      if arr.size > 0:
        ids.append(arr.astype(np.int32, copy=False))

  else:
    # Basic English word-level path
    assert tokenizer_fn is not None and vocab is not None
    for line in texts:
      toks = tokenizer_fn(line)
      if not toks:
        continue
      arr = vocab.encode_tokens(toks)
      if arr.size > 0:
        ids.append(arr.astype(np.int32, copy=False))

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


def make_epoch_factory(
        ds_obj: TextData,
        batch_size: int,
        rng: Optional[np.random.Generator] = None
):
    """
    Yields batches shaped like your previous collate_fn:
      - stack B sequences [B, T]
      - then transpose to [T, B]
      - targets flattened to [T*B]
    Returns a 0-arg factory. Each call -> one epoch iterator.
    If rng is provided, indices are shuffled once per epoch.
    """
    def factory():
      n = len(ds_obj)
      if rng is None:
        order = np.arange(n)
      else:
        order = rng.permutation(n)

      # same batching / shapes as _batch_iterator
      for i in range(0, n, batch_size):
        idxs = order[i:i + batch_size]
        xs = []
        ys = []
        for k in idxs:
          x, y = ds_obj[int(k)]
          xs.append(x)
          ys.append(y)
        xx = np.stack(xs, axis=0)  # [B, T]
        yy = np.stack(ys, axis=0)  # [B, T]
        xx = xx.transpose(1, 0).copy()  # [T, B]
        yy = yy.transpose(1, 0).reshape(-1).copy()  # [T*B]
        yield xx, yy

    return factory


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
        tokenizer: Optional[object] = None,
        eval_batch_size: int = 0,
        vocab: Optional[object] = None,
        basic_english_tokenizer: bool = False,
):
    """
    WikiText-103 loader with caching.
    • Defaults to GPT-2 BPE tokenizer (≈50 k vocab).
    • If `basic_english_tokenizer=True`, uses the older word-level pipeline.

    Returns:
        train_loader, val_loader, info
    """
    save_path.mkdir(parents=True, exist_ok=True)
    train_ids_path = save_path / "train.npy"
    val_ids_path   = save_path / "val.npy"
    vocab_path     = save_path / "wiki.vocab.pkl"

    # 1) Load HF dataset (raw text)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_texts = (x["text"] for x in ds["train"])
    val_texts   = (x["text"] for x in ds["validation"])

    # 2) Tokenization / IDs (cached)
    if not basic_english_tokenizer:
        # ---- GPT-2 branch (default) ----
        if tokenizer is None:
            tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        if not train_ids_path.exists():
            train_ids = _encode_split_to_ids(train_texts, hf_tokenizer=tokenizer)
            np.save(train_ids_path, train_ids)
        else:
            train_ids = np.load(train_ids_path, mmap_mode="r")

        if not val_ids_path.exists():
            val_ids = _encode_split_to_ids(val_texts, hf_tokenizer=tokenizer)
            np.save(val_ids_path, val_ids)
        else:
            val_ids = np.load(val_ids_path, mmap_mode="r")

        num_classes = tokenizer.vocab_size
        vocab_obj = None

    else:
        # ---- Basic English fallback ----
        if vocab is None:
            vocab_obj = build_or_load_vocab(save_path, (x["text"] for x in ds["train"]), basic_english)
        else:
            vocab_obj = vocab

        if not train_ids_path.exists():
            train_ids = _encode_split_to_ids(
              (x["text"] for x in ds["train"]),
              tokenizer_fn=basic_english,
              vocab=vocab_obj,
            )
            np.save(train_ids_path, train_ids)
        else:
            train_ids = np.load(train_ids_path, mmap_mode="r")

        if not val_ids_path.exists():
            val_ids = _encode_split_to_ids(
              (x["text"] for x in ds["validation"]),
              tokenizer_fn=basic_english,
              vocab=vocab_obj,
            )
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

    # 4) Build per-epoch factories
    rng = np.random.default_rng()  # or pass a seed from cfg if you want determinism

    train_epoch_factory = make_epoch_factory(train_dataset, batch_size, rng=rng)
    val_epoch_factory = make_epoch_factory(val_dataset, eval_batch_size or batch_size, rng=None)  # no shuffle for eval

    # Train: infinite stream with per-epoch shuffling
    train_loader = LoaderAsJaxIterator(train_epoch_factory, prefetch=4, repeat=True)

    # Eval: single pass, no shuffling
    eval_val_loader = LoaderAsJaxIterator(val_epoch_factory, prefetch=2, repeat=False)

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
      "tokenizer_type": "gpt2" if not basic_english_tokenizer else "basic_english",
    }

    return train_loader, eval_val_loader, info
