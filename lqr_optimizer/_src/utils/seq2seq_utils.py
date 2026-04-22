"""Seq2seq-specific runtime helpers kept separate from generic utils."""

import math
from collections.abc import Mapping, Sequence

import jax.numpy as jnp


SEQ2SEQ_TASK_KIND = "seq2seq_translation"
SEQ2SEQ_DEFAULT_PAD_ID = 1
_REQUIRED_MODEL_INIT_KWARGS = (
  "src_vocab_size",
  "tgt_vocab_size",
  "pad_id",
  "bos_id",
  "eos_id",
)


def unpack_seq2seq_inputs(x_batch):
  """Return `(src_tokens, prev_output_tokens)` from a seq2seq input tuple."""
  if not isinstance(x_batch, Sequence) or isinstance(x_batch, (str, bytes)):
    raise ValueError(
      "Seq2seq inputs must be a 2-item sequence `(src_tokens, prev_output_tokens)`."
    )
  if len(x_batch) != 2:
    raise ValueError(
      f"Seq2seq inputs must contain exactly two items, got {len(x_batch)}."
    )
  return x_batch[0], x_batch[1]


def validate_seq2seq_dataset_info(ds_info):
  """Validate and return the seq2seq-specific model kwargs from `ds_info`."""
  if not isinstance(ds_info, Mapping):
    raise ValueError("Seq2seq dataset info must be a mapping.")

  model_init_kwargs = ds_info.get("model_init_kwargs")
  if not isinstance(model_init_kwargs, Mapping):
    raise ValueError(
      "Seq2seq dataset info must define mapping-valued `model_init_kwargs`."
    )

  missing = [key for key in _REQUIRED_MODEL_INIT_KWARGS if key not in model_init_kwargs]
  if missing:
    raise ValueError(
      "Seq2seq dataset info is missing required `model_init_kwargs`: "
      + ", ".join(missing)
      + "."
    )

  normalized = dict(model_init_kwargs)
  for key in _REQUIRED_MODEL_INIT_KWARGS:
    try:
      normalized[key] = int(normalized[key])
    except (TypeError, ValueError) as exc:
      raise ValueError(
        f"Seq2seq dataset info field `{key}` must be an integer-compatible value."
      ) from exc

  if normalized["src_vocab_size"] <= 0 or normalized["tgt_vocab_size"] <= 0:
    raise ValueError("Seq2seq vocabulary sizes must be positive.")

  return normalized


def infer_seq2seq_target_lengths(prev_output_tokens, *, pad_id: int = SEQ2SEQ_DEFAULT_PAD_ID):
  prev_output_tokens = jnp.asarray(prev_output_tokens)
  if prev_output_tokens.ndim != 2:
    raise ValueError(
      "`prev_output_tokens` must be rank-2 to infer seq2seq target lengths."
    )
  lengths = jnp.sum(prev_output_tokens != int(pad_id), axis=1)
  return tuple(int(length) for length in lengths.tolist())


def seq2seq_target_offsets(target_lengths):
  offsets = [0]
  running = 0
  for length in target_lengths:
    running += int(length)
    offsets.append(running)
  return tuple(offsets)


def seq2seq_target_count(target_lengths, *, start: int = 0, size: int | None = None) -> int:
  stop = len(target_lengths) if size is None else start + int(size)
  return int(sum(int(length) for length in target_lengths[start:stop]))


def slice_seq2seq_flat_targets(targets, target_lengths, *, start: int, size: int):
  offsets = seq2seq_target_offsets(target_lengths)
  targets = jnp.asarray(targets)
  return targets[offsets[int(start)] : offsets[int(start + size)]]


def infer_seq2seq_batch_layout(batch, *, pad_id: int = SEQ2SEQ_DEFAULT_PAD_ID):
  x_batch, y_batch = batch
  src_tokens, prev_output_tokens = unpack_seq2seq_inputs(x_batch)
  src_tokens = jnp.asarray(src_tokens)
  prev_output_tokens = jnp.asarray(prev_output_tokens)
  y_batch = jnp.asarray(y_batch)

  if src_tokens.ndim != 2 or prev_output_tokens.ndim != 2:
    raise ValueError(
      "Seq2seq batches must provide rank-2 `src_tokens` and `prev_output_tokens`."
    )
  if y_batch.ndim != 1:
    raise ValueError("Seq2seq targets must be a flat rank-1 array.")

  target_lengths = infer_seq2seq_target_lengths(prev_output_tokens, pad_id=pad_id)
  expected_target_count = seq2seq_target_count(target_lengths)
  if int(y_batch.shape[0]) != expected_target_count:
    raise ValueError(
      "Seq2seq batch target length mismatch: expected "
      f"{expected_target_count} flattened targets, got {int(y_batch.shape[0])}."
    )

  return {
    "mode": SEQ2SEQ_TASK_KIND,
    "batch_axis": 0,
    "T": None,
    "pad_id": int(pad_id),
    "target_lengths": target_lengths,
    "max_target_length": int(max(target_lengths, default=0)),
    "average_target_length": float(
      seq2seq_target_count(target_lengths) / max(len(target_lengths), 1)
    ),
  }
