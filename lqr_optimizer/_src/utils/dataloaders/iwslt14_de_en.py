from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple

import numpy as np

from lqr_optimizer._src.utils.dataloaders.hf_loaders import LoaderAsJaxIterator
from lqr_optimizer._src.utils.seq2seq_utils import SEQ2SEQ_TASK_KIND


SPECIAL_TOKENS = ("<s>", "<pad>", "</s>", "<unk>")
EXPECTED_TEXT_FILES = (
  "train.de",
  "train.en",
  "valid.de",
  "valid.en",
  "test.de",
  "test.en",
  "code",
)


class FairseqStyleDictionary:
  """Minimal fairseq-compatible text dictionary without a fairseq dependency."""

  def __init__(self):
    self.symbols = list(SPECIAL_TOKENS)
    self.counts = [0] * len(self.symbols)
    self.indices = {symbol: index for index, symbol in enumerate(self.symbols)}
    self.nspecial = len(self.symbols)

  @property
  def bos_index(self) -> int:
    return 0

  @property
  def pad_index(self) -> int:
    return 1

  @property
  def eos_index(self) -> int:
    return 2

  @property
  def unk_index(self) -> int:
    return 3

  def __len__(self) -> int:
    return len(self.symbols)

  def add_symbol(self, token: str, count: int = 1) -> int:
    if token in self.indices:
      index = self.indices[token]
      self.counts[index] += int(count)
      return index
    index = len(self.symbols)
    self.indices[token] = index
    self.symbols.append(token)
    self.counts.append(int(count))
    return index

  def index(self, token: str) -> int:
    return self.indices.get(token, self.unk_index)

  def string(self, token_ids, *, remove_bpe=None, unk_string: str = "<unk>") -> str:
    words = []
    for token_id in np.asarray(token_ids, dtype=np.int32).tolist():
      token_id = int(token_id)
      if token_id == self.pad_index or token_id == self.bos_index:
        continue
      if token_id == self.eos_index:
        break
      if token_id == self.unk_index:
        token = unk_string
      elif 0 <= token_id < len(self.symbols):
        token = self.symbols[token_id]
      else:
        raise ValueError(f"Token id {token_id} is out of range for this dictionary.")
      words.append(token)
    sentence = " ".join(words)
    if remove_bpe:
      marker = "@@ " if remove_bpe is True else str(remove_bpe)
      sentence = sentence.replace(marker, "")
    return sentence

  def encode_line(self, line: str) -> np.ndarray:
    tokens = line.strip().split()
    ids = np.empty(len(tokens) + 1, dtype=np.int32)
    for idx, token in enumerate(tokens):
      ids[idx] = self.index(token)
    ids[len(tokens)] = self.eos_index
    return ids

  def pad_to_multiple(self, padding_factor: int) -> None:
    if padding_factor <= 1:
      return
    madeup_index = 0
    while len(self) % padding_factor != 0:
      token = f"madeupword{madeup_index:04d}"
      self.add_symbol(token, count=0)
      madeup_index += 1

  def save(self, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
      for token, count in zip(self.symbols[self.nspecial :], self.counts[self.nspecial :]):
        handle.write(f"{token} {count}\n")

  @classmethod
  def load(cls, path: Path) -> "FairseqStyleDictionary":
    dictionary = cls()
    with path.open("r", encoding="utf-8") as handle:
      for line in handle:
        token, count = line.rstrip("\n").rsplit(" ", 1)
        dictionary.add_symbol(token, count=int(count))
    return dictionary


class IndexedTokenStorage:
  def __init__(self, tokens: np.ndarray, offsets: np.ndarray):
    self.tokens = tokens
    self.offsets = offsets
    self.sizes = np.diff(offsets).astype(np.int32, copy=False)

  def __len__(self) -> int:
    return int(self.offsets.shape[0] - 1)

  def __getitem__(self, index: int) -> np.ndarray:
    start = int(self.offsets[index])
    stop = int(self.offsets[index + 1])
    return np.asarray(self.tokens[start:stop], dtype=np.int32)


class ParallelTextDataset:
  def __init__(self, src_storage: IndexedTokenStorage, tgt_storage: IndexedTokenStorage):
    if len(src_storage) != len(tgt_storage):
      raise ValueError("Source and target splits must contain the same number of examples.")
    self.src = src_storage
    self.tgt = tgt_storage
    self.src_sizes = src_storage.sizes
    self.tgt_sizes = tgt_storage.sizes

  def __len__(self) -> int:
    return len(self.src)


@dataclass(frozen=True)
class TranslationBatchSpec:
  indices: np.ndarray
  padded_batch_size: int
  padded_src_length: int
  padded_tgt_length: int


def prepare_local_seq2seq_dataset(name: str):
  if name == "iwslt14_de_en":
    return load_iwslt14_de_en
  raise ValueError(f"{name} dataset is not supported")


def _require_extracted_dataset(dataset_dir: Path, source_lang: str, target_lang: str) -> None:
  if not dataset_dir.exists():
    raise FileNotFoundError(f"Expected extracted dataset directory at {dataset_dir}")
  required = {name.replace(".de", f".{source_lang}").replace(".en", f".{target_lang}") for name in EXPECTED_TEXT_FILES}
  missing = [name for name in required if not (dataset_dir / name).exists()]
  if missing:
    raise FileNotFoundError(
      "Extracted IWSLT14 dataset is missing required files: " + ", ".join(sorted(missing))
    )


def _build_dictionary(lines: Iterable[str], padding_factor: int) -> FairseqStyleDictionary:
  counter: Counter[str] = Counter()
  for line in lines:
    counter.update(line.strip().split())
  dictionary = FairseqStyleDictionary()
  for token, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
    dictionary.add_symbol(token, count=count)
  dictionary.pad_to_multiple(padding_factor)
  return dictionary


def _encode_lines(lines: Sequence[str], dictionary: FairseqStyleDictionary) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
  encoded = [dictionary.encode_line(line) for line in lines]
  lengths = np.asarray([tokens.shape[0] for tokens in encoded], dtype=np.int64)
  offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
  offsets[1:] = np.cumsum(lengths)
  if encoded:
    flat_tokens = np.concatenate(encoded).astype(np.int32, copy=False)
  else:
    flat_tokens = np.empty((0,), dtype=np.int32)
  stats = {
    "num_examples": int(len(encoded)),
    "token_count": int(lengths.sum()),
    "max_length": int(lengths.max()) if lengths.size else 0,
  }
  return flat_tokens, offsets, stats


def _save_numpy(path: Path, array: np.ndarray) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  np.save(path, array)


def _cache_is_complete(cache_dir: Path, source_lang: str, target_lang: str) -> bool:
  required = [
    cache_dir / "metadata.json",
    cache_dir / f"dict.{source_lang}.txt",
    cache_dir / f"dict.{target_lang}.txt",
  ]
  for split in ("train", "valid", "test"):
    required.extend(
      [
        cache_dir / f"{split}.{source_lang}.tokens.npy",
        cache_dir / f"{split}.{source_lang}.offsets.npy",
        cache_dir / f"{split}.{target_lang}.tokens.npy",
        cache_dir / f"{split}.{target_lang}.offsets.npy",
      ]
    )
  return all(path.exists() for path in required)


def _read_lines(path: Path) -> list[str]:
  with path.open("r", encoding="utf-8") as handle:
    return [line.rstrip("\n") for line in handle]


def _build_numeric_cache(
    dataset_dir: Path,
    cache_dir: Path,
    *,
    source_lang: str,
    target_lang: str,
    padding_factor: int,
) -> Dict[str, object]:
  cache_dir.mkdir(parents=True, exist_ok=True)
  train_src_lines = _read_lines(dataset_dir / f"train.{source_lang}")
  train_tgt_lines = _read_lines(dataset_dir / f"train.{target_lang}")
  dictionaries = {
    source_lang: _build_dictionary(train_src_lines, padding_factor),
    target_lang: _build_dictionary(train_tgt_lines, padding_factor),
  }
  dictionaries[source_lang].save(cache_dir / f"dict.{source_lang}.txt")
  dictionaries[target_lang].save(cache_dir / f"dict.{target_lang}.txt")

  split_stats: Dict[str, Dict[str, int]] = {}
  for split in ("train", "valid", "test"):
    src_lines = _read_lines(dataset_dir / f"{split}.{source_lang}")
    tgt_lines = _read_lines(dataset_dir / f"{split}.{target_lang}")
    if len(src_lines) != len(tgt_lines):
      raise ValueError(f"Split {split} has mismatched source and target line counts.")
    src_tokens, src_offsets, src_stats = _encode_lines(src_lines, dictionaries[source_lang])
    tgt_tokens, tgt_offsets, tgt_stats = _encode_lines(tgt_lines, dictionaries[target_lang])
    _save_numpy(cache_dir / f"{split}.{source_lang}.tokens.npy", src_tokens)
    _save_numpy(cache_dir / f"{split}.{source_lang}.offsets.npy", src_offsets)
    _save_numpy(cache_dir / f"{split}.{target_lang}.tokens.npy", tgt_tokens)
    _save_numpy(cache_dir / f"{split}.{target_lang}.offsets.npy", tgt_offsets)
    split_stats[split] = {
      "num_examples": int(len(src_lines)),
      "src_token_count": src_stats["token_count"],
      "tgt_token_count": tgt_stats["token_count"],
      "max_src_length": src_stats["max_length"],
      "max_tgt_length": tgt_stats["max_length"],
    }

  metadata = {
    "task_kind": SEQ2SEQ_TASK_KIND,
    "source_lang": source_lang,
    "target_lang": target_lang,
    "padding_factor": int(padding_factor),
    "special_tokens": {
      "bos": SPECIAL_TOKENS[0],
      "pad": SPECIAL_TOKENS[1],
      "eos": SPECIAL_TOKENS[2],
      "unk": SPECIAL_TOKENS[3],
      "bos_id": dictionaries[source_lang].bos_index,
      "pad_id": dictionaries[source_lang].pad_index,
      "eos_id": dictionaries[source_lang].eos_index,
      "unk_id": dictionaries[source_lang].unk_index,
    },
    "vocab_sizes": {
      source_lang: len(dictionaries[source_lang]),
      target_lang: len(dictionaries[target_lang]),
    },
    "splits": split_stats,
  }
  (cache_dir / "metadata.json").write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return metadata


def _load_numeric_cache(
    dataset_dir: Path,
    cache_dir: Path,
    *,
    source_lang: str,
    target_lang: str,
    padding_factor: int,
) -> Dict[str, object]:
  if not _cache_is_complete(cache_dir, source_lang, target_lang):
    return _build_numeric_cache(
      dataset_dir,
      cache_dir,
      source_lang=source_lang,
      target_lang=target_lang,
      padding_factor=padding_factor,
    )
  return json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))


def _load_storage(cache_dir: Path, split: str, lang: str) -> IndexedTokenStorage:
  tokens = np.load(cache_dir / f"{split}.{lang}.tokens.npy", mmap_mode="r")
  offsets = np.load(cache_dir / f"{split}.{lang}.offsets.npy", mmap_mode="r")
  return IndexedTokenStorage(tokens=np.asarray(tokens, dtype=np.int32), offsets=np.asarray(offsets, dtype=np.int64))


def load_iwslt14_de_en_dictionaries(
    *,
    dataset_dir: Path,
    numeric_cache_dir: Optional[Path] = None,
    source_lang: str = "de",
    target_lang: str = "en",
    padding_factor: int = 8,
):
  dataset_dir = Path(dataset_dir)
  cache_dir = Path(numeric_cache_dir) if numeric_cache_dir is not None else dataset_dir / ".llqr_numeric_cache"
  _require_extracted_dataset(dataset_dir, source_lang=source_lang, target_lang=target_lang)
  metadata = _load_numeric_cache(
    dataset_dir,
    cache_dir,
    source_lang=source_lang,
    target_lang=target_lang,
    padding_factor=padding_factor,
  )
  return {
    "metadata": metadata,
    "cache_dir": cache_dir,
    "source_dictionary": FairseqStyleDictionary.load(cache_dir / f"dict.{source_lang}.txt"),
    "target_dictionary": FairseqStyleDictionary.load(cache_dir / f"dict.{target_lang}.txt"),
  }


def _ordered_indices(
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    *,
    shuffle: bool,
    rng: Optional[np.random.Generator],
) -> np.ndarray:
  if shuffle:
    if rng is None:
      raise ValueError("An RNG is required when shuffle=True.")
    indices = rng.permutation(len(src_sizes)).astype(np.int64)
  else:
    indices = np.arange(len(src_sizes), dtype=np.int64)
  indices = indices[np.argsort(tgt_sizes[indices], kind="mergesort")]
  indices = indices[np.argsort(src_sizes[indices], kind="mergesort")]
  return indices


def _batch_indices_by_max_tokens(
    ordered_indices: np.ndarray,
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    *,
    max_tokens: int,
) -> list[np.ndarray]:
  batches = []
  current: list[int] = []
  current_max_tokens = 0
  for raw_index in ordered_indices:
    index = int(raw_index)
    sample_tokens = max(int(src_sizes[index]), int(tgt_sizes[index]))
    proposed_max_tokens = max(current_max_tokens, sample_tokens)
    proposed_batch_size = len(current) + 1
    if current and proposed_batch_size * proposed_max_tokens > int(max_tokens):
      batches.append(np.asarray(current, dtype=np.int64))
      current = [index]
      current_max_tokens = sample_tokens
    else:
      current.append(index)
      current_max_tokens = proposed_max_tokens
  if current:
    batches.append(np.asarray(current, dtype=np.int64))
  return batches


def _round_up_to_multiple(value: int, multiple: int) -> int:
  if multiple <= 1:
    return int(value)
  return int(((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple))


def _build_dynamic_batch_specs_by_max_tokens(
    ordered_indices: np.ndarray,
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    *,
    max_tokens: int,
) -> list[TranslationBatchSpec]:
  batch_specs = []
  for batch_indices in _batch_indices_by_max_tokens(
      ordered_indices,
      src_sizes,
      tgt_sizes,
      max_tokens=max_tokens,
  ):
    batch_src_sizes = src_sizes[batch_indices]
    batch_tgt_sizes = tgt_sizes[batch_indices]
    batch_specs.append(
      TranslationBatchSpec(
        indices=np.asarray(batch_indices, dtype=np.int64),
        padded_batch_size=int(batch_indices.shape[0]),
        padded_src_length=int(batch_src_sizes.max()),
        padded_tgt_length=int(batch_tgt_sizes.max()),
      )
    )
  return batch_specs


def _build_static_shape_bucketed_batch_specs_by_max_tokens(
    ordered_indices: np.ndarray,
    src_sizes: np.ndarray,
    tgt_sizes: np.ndarray,
    *,
    max_tokens: int,
    shape_bucket_multiple: int,
) -> list[TranslationBatchSpec]:
  bucket_multiple = max(1, int(shape_bucket_multiple))
  batch_specs: list[TranslationBatchSpec] = []
  current: list[int] = []
  current_src_bucket = 0
  current_tgt_bucket = 0
  current_capacity = 0

  def flush_current() -> None:
    if not current:
      return
    batch_specs.append(
      TranslationBatchSpec(
        indices=np.asarray(current, dtype=np.int64),
        padded_batch_size=int(current_capacity),
        padded_src_length=int(current_src_bucket),
        padded_tgt_length=int(current_tgt_bucket),
      )
    )

  for raw_index in ordered_indices:
    index = int(raw_index)
    src_bucket = _round_up_to_multiple(int(src_sizes[index]), bucket_multiple)
    tgt_bucket = _round_up_to_multiple(int(tgt_sizes[index]), bucket_multiple)
    bucket_token_size = max(src_bucket, tgt_bucket)
    bucket_capacity = max(1, int(max_tokens) // int(bucket_token_size))

    bucket_changed = (
      current
      and (src_bucket != current_src_bucket or tgt_bucket != current_tgt_bucket)
    )
    batch_full = current and (len(current) >= current_capacity)
    if bucket_changed or batch_full:
      flush_current()
      current = []
      current_src_bucket = 0
      current_tgt_bucket = 0
      current_capacity = 0

    if not current:
      current_src_bucket = src_bucket
      current_tgt_bucket = tgt_bucket
      current_capacity = bucket_capacity

    current.append(index)

  flush_current()
  return batch_specs


def _batch_indices_by_sentences(ordered_indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
  batches = []
  for start in range(0, len(ordered_indices), int(batch_size)):
    batches.append(np.asarray(ordered_indices[start:start + int(batch_size)], dtype=np.int64))
  return batches


def _collate_translation_batch(
    dataset: ParallelTextDataset,
    batch_spec: np.ndarray | TranslationBatchSpec,
    *,
    pad_id: int,
    eos_id: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
  if isinstance(batch_spec, TranslationBatchSpec):
    indices = batch_spec.indices
    padded_batch_size = int(batch_spec.padded_batch_size)
    padded_src_length = int(batch_spec.padded_src_length)
    padded_tgt_length = int(batch_spec.padded_tgt_length)
  else:
    indices = np.asarray(batch_spec, dtype=np.int64)
    src_lengths = dataset.src_sizes[indices]
    tgt_lengths = dataset.tgt_sizes[indices]
    padded_batch_size = int(indices.shape[0])
    padded_src_length = int(src_lengths.max())
    padded_tgt_length = int(tgt_lengths.max())

  src_examples = [dataset.src[int(index)] for index in indices]
  tgt_examples = [dataset.tgt[int(index)] for index in indices]
  src_lengths = np.asarray([example.shape[0] for example in src_examples], dtype=np.int32)
  sort_order = np.argsort(-src_lengths, kind="mergesort")
  src_examples = [src_examples[int(index)] for index in sort_order]
  tgt_examples = [tgt_examples[int(index)] for index in sort_order]

  batch_size = len(src_examples)
  max_src_length = max(example.shape[0] for example in src_examples)
  max_tgt_length = max(example.shape[0] for example in tgt_examples)
  if padded_batch_size < batch_size:
    raise ValueError("Padded batch size cannot be smaller than the number of examples.")
  if padded_src_length < max_src_length:
    raise ValueError("Padded source length cannot be smaller than the batch maximum.")
  if padded_tgt_length < max_tgt_length:
    raise ValueError("Padded target length cannot be smaller than the batch maximum.")
  src_batch = np.full((padded_batch_size, padded_src_length), pad_id, dtype=np.int32)
  prev_output_batch = np.full((padded_batch_size, padded_tgt_length), pad_id, dtype=np.int32)
  flat_targets = []

  for row_index, (src_tokens, tgt_tokens) in enumerate(zip(src_examples, tgt_examples)):
    src_batch[row_index, -src_tokens.shape[0] :] = src_tokens
    prev_output = np.empty_like(tgt_tokens)
    prev_output[0] = int(eos_id)
    if tgt_tokens.shape[0] > 1:
      prev_output[1:] = tgt_tokens[:-1]
    prev_output_batch[row_index, : prev_output.shape[0]] = prev_output
    flat_targets.append(np.asarray(tgt_tokens, dtype=np.int32))

  flat_targets_array = np.concatenate(flat_targets).astype(np.int32, copy=False)
  padded_target_count = int(padded_batch_size * padded_tgt_length)
  y_batch = np.full((padded_target_count,), pad_id, dtype=np.int32)
  y_batch[: flat_targets_array.shape[0]] = flat_targets_array
  return (src_batch, prev_output_batch), y_batch


def _make_epoch_factory(
    dataset: ParallelTextDataset,
    *,
    pad_id: int,
    eos_id: int,
    max_tokens: Optional[int] = None,
    batch_size: Optional[int] = None,
    shuffle: bool,
    rng: Optional[np.random.Generator],
    static_shape_bucketing: bool = False,
    shape_bucket_multiple: int = 8,
) -> Callable[[], Iterator[Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]]:
  if (max_tokens is None) == (batch_size is None):
    raise ValueError("Specify exactly one of max_tokens or batch_size.")

  def factory():
    ordered = _ordered_indices(dataset.src_sizes, dataset.tgt_sizes, shuffle=shuffle, rng=rng)
    if max_tokens is not None:
      if static_shape_bucketing:
        batches = _build_static_shape_bucketed_batch_specs_by_max_tokens(
          ordered,
          dataset.src_sizes,
          dataset.tgt_sizes,
          max_tokens=max_tokens,
          shape_bucket_multiple=shape_bucket_multiple,
        )
      else:
        batches = _build_dynamic_batch_specs_by_max_tokens(
          ordered,
          dataset.src_sizes,
          dataset.tgt_sizes,
          max_tokens=max_tokens,
        )
    else:
      batches = _batch_indices_by_sentences(ordered, batch_size=batch_size)
    for batch_indices in batches:
      yield _collate_translation_batch(dataset, batch_indices, pad_id=pad_id, eos_id=eos_id)

  return factory


def load_iwslt14_de_en(
    *,
    dataset_dir: Path,
    numeric_cache_dir: Optional[Path] = None,
    max_tokens: int,
    eval_batch_size: int,
    prefetch: int = 4,
    eval_prefetch: int = 2,
    static_shape_bucketing: bool = True,
    shape_bucket_multiple: int = 8,
    source_lang: str = "de",
    target_lang: str = "en",
    padding_factor: int = 8,
):
  dataset_dir = Path(dataset_dir)
  cache_dir = Path(numeric_cache_dir) if numeric_cache_dir is not None else dataset_dir / ".llqr_numeric_cache"
  _require_extracted_dataset(dataset_dir, source_lang=source_lang, target_lang=target_lang)
  metadata = _load_numeric_cache(
    dataset_dir,
    cache_dir,
    source_lang=source_lang,
    target_lang=target_lang,
    padding_factor=padding_factor,
  )

  train_dataset = ParallelTextDataset(
    _load_storage(cache_dir, "train", source_lang),
    _load_storage(cache_dir, "train", target_lang),
  )
  valid_dataset = ParallelTextDataset(
    _load_storage(cache_dir, "valid", source_lang),
    _load_storage(cache_dir, "valid", target_lang),
  )

  rng = np.random.default_rng()
  pad_id = int(metadata["special_tokens"]["pad_id"])
  eos_id = int(metadata["special_tokens"]["eos_id"])

  train_epoch_factory = _make_epoch_factory(
    train_dataset,
    pad_id=pad_id,
    eos_id=eos_id,
    max_tokens=int(max_tokens),
    shuffle=True,
    rng=rng,
    static_shape_bucketing=bool(static_shape_bucketing),
    shape_bucket_multiple=int(shape_bucket_multiple),
  )
  valid_epoch_factory = _make_epoch_factory(
    valid_dataset,
    pad_id=pad_id,
    eos_id=eos_id,
    batch_size=int(eval_batch_size),
    shuffle=False,
    rng=None,
  )

  train_loader = LoaderAsJaxIterator(train_epoch_factory, prefetch=int(prefetch), repeat=True)
  valid_loader = LoaderAsJaxIterator(valid_epoch_factory, prefetch=int(eval_prefetch), repeat=False)

  train_order = _ordered_indices(train_dataset.src_sizes, train_dataset.tgt_sizes, shuffle=False, rng=None)
  if static_shape_bucketing:
    train_batch_specs = _build_static_shape_bucketed_batch_specs_by_max_tokens(
      train_order,
      train_dataset.src_sizes,
      train_dataset.tgt_sizes,
      max_tokens=int(max_tokens),
      shape_bucket_multiple=int(shape_bucket_multiple),
    )
  else:
    train_batch_specs = _build_dynamic_batch_specs_by_max_tokens(
      train_order,
      train_dataset.src_sizes,
      train_dataset.tgt_sizes,
      max_tokens=int(max_tokens),
    )
  train_shape_signatures = {
    (
      int(batch_spec.padded_batch_size),
      int(batch_spec.padded_src_length),
      int(batch_spec.padded_tgt_length),
    )
    for batch_spec in train_batch_specs
  }
  ds_info = {
    "num_classes": int(metadata["vocab_sizes"][target_lang]),
    "ds_size": int(metadata["splits"]["train"]["num_examples"]),
    "test_ds_size": int(metadata["splits"]["valid"]["num_examples"]),
    "task_kind": SEQ2SEQ_TASK_KIND,
    "model_init_kwargs": {
      "src_vocab_size": int(metadata["vocab_sizes"][source_lang]),
      "tgt_vocab_size": int(metadata["vocab_sizes"][target_lang]),
      "pad_id": pad_id,
      "bos_id": int(metadata["special_tokens"]["bos_id"]),
      "eos_id": eos_id,
    },
    "runner_contract": {
      "train_microbatches_per_epoch": int(len(train_batch_specs)),
      "train_eval_target_count": int(metadata["splits"]["train"]["tgt_token_count"]),
      "test_eval_target_count": int(metadata["splits"]["valid"]["tgt_token_count"]),
    },
    "batch_shape_contract": {
      "static_shape_bucketing": bool(static_shape_bucketing),
      "shape_bucket_multiple": int(shape_bucket_multiple),
      "train_shape_signature_count": int(len(train_shape_signatures)),
      "train_shape_signatures": [
        {
          "batch_size": int(signature[0]),
          "src_length": int(signature[1]),
          "tgt_length": int(signature[2]),
        }
        for signature in sorted(train_shape_signatures)
      ],
    },
    "source_lang": source_lang,
    "target_lang": target_lang,
  }
  return train_loader, valid_loader, ds_info


def load_iwslt14_de_en_eval_split(
    *,
    dataset_dir: Path,
    split: str,
    numeric_cache_dir: Optional[Path] = None,
    eval_batch_size: int,
    eval_prefetch: int = 2,
    source_lang: str = "de",
    target_lang: str = "en",
    padding_factor: int = 8,
):
  if split not in ("valid", "test"):
    raise ValueError(f"Expected split='valid' or split='test', got {split!r}.")

  dictionary_bundle = load_iwslt14_de_en_dictionaries(
    dataset_dir=dataset_dir,
    numeric_cache_dir=numeric_cache_dir,
    source_lang=source_lang,
    target_lang=target_lang,
    padding_factor=padding_factor,
  )
  cache_dir = dictionary_bundle["cache_dir"]
  metadata = dictionary_bundle["metadata"]

  dataset = ParallelTextDataset(
    _load_storage(cache_dir, split, source_lang),
    _load_storage(cache_dir, split, target_lang),
  )
  pad_id = int(metadata["special_tokens"]["pad_id"])
  eos_id = int(metadata["special_tokens"]["eos_id"])
  epoch_factory = _make_epoch_factory(
    dataset,
    pad_id=pad_id,
    eos_id=eos_id,
    batch_size=int(eval_batch_size),
    shuffle=False,
    rng=None,
  )
  loader = LoaderAsJaxIterator(epoch_factory, prefetch=int(eval_prefetch), repeat=False)
  split_info = {
    "split": split,
    "num_examples": int(metadata["splits"][split]["num_examples"]),
    "source_lang": source_lang,
    "target_lang": target_lang,
    "pad_id": pad_id,
    "bos_id": int(metadata["special_tokens"]["bos_id"]),
    "eos_id": eos_id,
    "source_dictionary": dictionary_bundle["source_dictionary"],
    "target_dictionary": dictionary_bundle["target_dictionary"],
  }
  return loader, split_info
