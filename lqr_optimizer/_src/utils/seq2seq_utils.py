"""Seq2seq-specific runtime helpers kept separate from generic utils."""

from collections.abc import Mapping, Sequence


SEQ2SEQ_TASK_KIND = "seq2seq_translation"
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
