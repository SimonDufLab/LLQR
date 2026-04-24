"""Seq2seq-specific runtime helpers kept separate from generic utils."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


SEQ2SEQ_TASK_KIND = "seq2seq_translation"
SEQ2SEQ_DEFAULT_PAD_ID = 1
_REQUIRED_MODEL_INIT_KWARGS = (
  "src_vocab_size",
  "tgt_vocab_size",
  "pad_id",
  "bos_id",
  "eos_id",
)


@dataclass(frozen=True)
class Seq2SeqBeamCandidate:
  tokens: tuple[int, ...]
  score: float
  finished: bool


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


def recover_seq2seq_target_sequences(prev_output_tokens, targets, *, pad_id: int = SEQ2SEQ_DEFAULT_PAD_ID):
  target_lengths = infer_seq2seq_target_lengths(prev_output_tokens, pad_id=pad_id)
  offsets = seq2seq_target_offsets(target_lengths)
  targets = np.asarray(targets, dtype=np.int32)
  return [
    np.asarray(targets[offsets[index] : offsets[index + 1]], dtype=np.int32)
    for index in range(len(target_lengths))
  ]


def pad_and_concatenate_seq2seq_input_batches(input_batches, *, pad_id: int):
  normalized_batches = []
  max_src_length = 0
  max_tgt_length = 0
  for inputs in input_batches:
    src_tokens, prev_output_tokens = unpack_seq2seq_inputs(inputs)
    src_tokens = np.asarray(src_tokens, dtype=np.int32)
    prev_output_tokens = np.asarray(prev_output_tokens, dtype=np.int32)
    normalized_batches.append((src_tokens, prev_output_tokens))
    max_src_length = max(max_src_length, int(src_tokens.shape[1]))
    max_tgt_length = max(max_tgt_length, int(prev_output_tokens.shape[1]))

  padded_src_batches = []
  padded_tgt_batches = []
  for src_tokens, prev_output_tokens in normalized_batches:
    if src_tokens.shape[1] < max_src_length:
      src_tokens = np.pad(
        src_tokens,
        ((0, 0), (max_src_length - src_tokens.shape[1], 0)),
        constant_values=int(pad_id),
      )
    if prev_output_tokens.shape[1] < max_tgt_length:
      prev_output_tokens = np.pad(
        prev_output_tokens,
        ((0, 0), (0, max_tgt_length - prev_output_tokens.shape[1])),
        constant_values=int(pad_id),
      )
    padded_src_batches.append(src_tokens)
    padded_tgt_batches.append(prev_output_tokens)

  return (
    jnp.asarray(np.concatenate(padded_src_batches, axis=0), dtype=jnp.int32),
    jnp.asarray(np.concatenate(padded_tgt_batches, axis=0), dtype=jnp.int32),
  )


def strip_seq2seq_tokens(token_ids, *, pad_id: int, bos_id: int, eos_id: int) -> tuple[int, ...]:
  stripped = []
  for token_id in np.asarray(token_ids, dtype=np.int32).tolist():
    token_id = int(token_id)
    if token_id == pad_id:
      continue
    if token_id == bos_id and not stripped:
      continue
    if token_id == eos_id:
      break
    stripped.append(token_id)
  return tuple(stripped)


def seq2seq_source_length(src_tokens, *, pad_id: int, eos_id: int) -> int:
  src_tokens = np.asarray(src_tokens, dtype=np.int32)
  return int(np.sum((src_tokens != int(pad_id)) & (src_tokens != int(eos_id))))


def remove_bpe_markers(text: str, marker) -> str:
  if marker in (None, False, ""):
    return text
  resolved_marker = "@@ " if marker is True else str(marker)
  return text.replace(resolved_marker, "")


def build_translation_detokenizer(detok: str | None, *, target_lang: str):
  if detok in (None, "space", "none"):
    return lambda text: text
  if detok != "moses":
    raise ValueError(
      f"Unsupported translation detokenizer '{detok}'. Expected 'moses' or 'space'."
    )
  try:
    from sacremoses import MosesDetokenizer
  except ImportError as exc:
    raise ImportError(
      "Translation BLEU with detok='moses' requires sacremoses. "
      "Install it with `uv add sacremoses`."
    ) from exc
  detokenizer = MosesDetokenizer(lang=target_lang)
  return lambda text: detokenizer.detokenize(text.split())


def score_translation_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
  try:
    import sacrebleu
  except ImportError as exc:
    raise ImportError(
      "Translation BLEU scoring requires sacrebleu. Install it with `uv add sacrebleu`."
    ) from exc
  return float(sacrebleu.corpus_bleu(list(hypotheses), [list(references)]).score)


def decode_seq2seq_text(dictionary, token_ids, *, remove_bpe, detokenize: Callable[[str], str], unk_string: str) -> str:
  text = dictionary.string(token_ids, remove_bpe=remove_bpe, unk_string=unk_string)
  return detokenize(text)


def translation_next_log_probs(next_log_probs_fn, src_tokens, prev_output_tokens):
  src_tokens = jnp.asarray(src_tokens, dtype=jnp.int32)
  prev_output_tokens = jnp.asarray(prev_output_tokens, dtype=jnp.int32)
  return jnp.asarray(next_log_probs_fn(src_tokens, prev_output_tokens))


def generate_seq2seq_beam_search(
    *,
    next_log_probs_fn,
    src_tokens,
    beam_size: int,
    max_len_a: float,
    max_len_b: int,
    pad_id: int,
    eos_id: int,
    max_target_positions: int,
):
  if beam_size <= 0:
    raise ValueError(f"beam_size must be positive, got {beam_size}.")

  src_tokens = np.asarray(src_tokens, dtype=np.int32)
  src_length = seq2seq_source_length(src_tokens, pad_id=pad_id, eos_id=eos_id)
  max_generated_length = int(max_len_a * src_length + max_len_b)
  max_generated_length = max(1, min(int(max_target_positions) - 1, max_generated_length))
  if max_generated_length <= 0:
    raise ValueError("max_target_positions must be at least 2 for seq2seq generation.")

  beams = [Seq2SeqBeamCandidate(tokens=(int(eos_id),), score=0.0, finished=False)]
  for _ in range(max_generated_length):
    active = [beam for beam in beams if not beam.finished]
    finished = [beam for beam in beams if beam.finished]
    if not active:
      break

    repeated_src = np.repeat(src_tokens[None, :], len(active), axis=0)
    active_prev_output = np.asarray([beam.tokens for beam in active], dtype=np.int32)
    step_log_probs = np.asarray(
      translation_next_log_probs(next_log_probs_fn, repeated_src, active_prev_output),
      dtype=np.float32,
    )

    candidates = list(finished)
    for beam_index, beam in enumerate(active):
      top_indices = np.argsort(-step_log_probs[beam_index])[:beam_size]
      for token_id in top_indices.tolist():
        token_id = int(token_id)
        candidates.append(
          Seq2SeqBeamCandidate(
            tokens=beam.tokens + (token_id,),
            score=float(beam.score + step_log_probs[beam_index, token_id]),
            finished=bool(token_id == int(eos_id)),
          )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    beams = candidates[:beam_size]
    if all(candidate.finished for candidate in beams):
      break

  completed = [beam for beam in beams if beam.finished]
  best = max(completed or beams, key=lambda candidate: candidate.score)
  return np.asarray(best.tokens[1:], dtype=np.int32)


def evaluate_translation_generation(
    *,
    dataloader,
    next_log_probs_fn,
    target_dictionary,
    target_lang: str,
    beam_size: int,
    max_len_a: float,
    max_len_b: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    max_target_positions: int,
    remove_bpe,
    detok: str | None,
):
  if hasattr(dataloader, "reset"):
    dataloader.reset()

  detokenize = build_translation_detokenizer(detok, target_lang=target_lang)
  hypotheses = []
  references = []
  sample_hypothesis = None
  sample_reference = None

  for x_batch, y_batch in dataloader:
    src_tokens, prev_output_tokens = unpack_seq2seq_inputs(x_batch)
    target_sequences = recover_seq2seq_target_sequences(prev_output_tokens, y_batch, pad_id=pad_id)
    src_tokens = np.asarray(src_tokens, dtype=np.int32)
    for example_index, reference_tokens in enumerate(target_sequences):
      hypothesis_tokens = generate_seq2seq_beam_search(
        next_log_probs_fn=next_log_probs_fn,
        src_tokens=src_tokens[example_index],
        beam_size=beam_size,
        max_len_a=max_len_a,
        max_len_b=max_len_b,
        pad_id=pad_id,
        eos_id=eos_id,
        max_target_positions=max_target_positions,
      )
      hypothesis = decode_seq2seq_text(
        target_dictionary,
        hypothesis_tokens,
        remove_bpe=remove_bpe,
        detokenize=detokenize,
        unk_string="UNKNOWNTOKENINHYP",
      )
      reference = decode_seq2seq_text(
        target_dictionary,
        reference_tokens,
        remove_bpe=remove_bpe,
        detokenize=detokenize,
        unk_string="UNKNOWNTOKENINREF",
      )
      hypotheses.append(hypothesis)
      references.append(reference)
      if sample_hypothesis is None:
        sample_hypothesis = hypothesis
        sample_reference = reference

  return {
    "bleu": score_translation_bleu(hypotheses, references),
    "hypotheses": hypotheses,
    "references": references,
    "sample_hypothesis": sample_hypothesis,
    "sample_reference": sample_reference,
    "num_examples": len(hypotheses),
  }


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
  observed_target_count = int(y_batch.shape[0])
  if observed_target_count < expected_target_count:
    raise ValueError(
      "Seq2seq batch target length mismatch: expected "
      f"at least {expected_target_count} flattened targets, got {observed_target_count}."
    )
  if observed_target_count > expected_target_count:
    padded_tail = y_batch[expected_target_count:]
    if int(jnp.count_nonzero(padded_tail != int(pad_id))) != 0:
      raise ValueError(
        "Seq2seq batch target padding must use the dataset pad_id when the "
        "flattened target vector is compilation-padded."
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
