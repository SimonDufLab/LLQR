"""Seq2seq-specific runtime helpers kept separate from generic utils."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


SEQ2SEQ_TASK_KIND = "seq2seq_translation"
SEQ2SEQ_DEFAULT_PAD_ID = 1
_FAIRSEQ_DEFAULT_MIN_LEN = 1
_FAIRSEQ_DEFAULT_LEN_PENALTY = 1.0
_REQUIRED_MODEL_INIT_KWARGS = (
  "src_vocab_size",
  "tgt_vocab_size",
  "pad_id",
  "bos_id",
  "eos_id",
)
_REQUIRED_PRECONDITIONER_SHAPE_KWARGS = (
  "canonical_src_length",
  "canonical_tgt_length",
  "pad_id",
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


def validate_seq2seq_preconditioner_shape_contract(shape_contract):
  """Validate and normalize the seq2seq preconditioner shape contract."""
  if not isinstance(shape_contract, Mapping):
    raise ValueError("Seq2seq preconditioner shape contract must be a mapping.")

  missing = [
    key for key in _REQUIRED_PRECONDITIONER_SHAPE_KWARGS if key not in shape_contract
  ]
  if missing:
    raise ValueError(
      "Seq2seq preconditioner shape contract is missing required fields: "
      + ", ".join(missing)
      + "."
    )

  normalized = dict(shape_contract)
  normalized.setdefault("mode", SEQ2SEQ_TASK_KIND)
  if normalized["mode"] != SEQ2SEQ_TASK_KIND:
    raise ValueError(
      "Seq2seq preconditioner shape contract must target mode "
      f"{SEQ2SEQ_TASK_KIND!r}, got {normalized['mode']!r}."
    )
  for key in _REQUIRED_PRECONDITIONER_SHAPE_KWARGS:
    try:
      normalized[key] = int(normalized[key])
    except (TypeError, ValueError) as exc:
      raise ValueError(
        f"Seq2seq preconditioner shape contract field `{key}` must be integer-compatible."
      ) from exc
  if normalized["canonical_src_length"] <= 0 or normalized["canonical_tgt_length"] <= 0:
    raise ValueError("Seq2seq canonical source/target lengths must be positive.")
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


def pad_seq2seq_datapoint_to_static_shape(
    datapoint,
    *,
    padded_batch_size: int,
    padded_src_length: int,
    padded_tgt_length: int,
    pad_id: int = SEQ2SEQ_DEFAULT_PAD_ID,
):
  """Pad a seq2seq datapoint to a fixed batch/source/target signature."""
  if int(padded_batch_size) <= 0:
    raise ValueError("`padded_batch_size` must be positive.")
  if int(padded_src_length) <= 0 or int(padded_tgt_length) <= 0:
    raise ValueError("Seq2seq static source/target lengths must be positive.")

  layout = infer_seq2seq_batch_layout(datapoint, pad_id=pad_id)
  inputs, targets = datapoint
  src_tokens, prev_output_tokens = unpack_seq2seq_inputs(inputs)
  src_tokens = jnp.asarray(src_tokens)
  prev_output_tokens = jnp.asarray(prev_output_tokens)
  targets = jnp.asarray(targets)

  live_batch_size = int(sum(int(length) > 0 for length in layout["target_lengths"]))
  live_target_count = seq2seq_target_count(layout["target_lengths"])
  if live_batch_size > int(padded_batch_size):
    raise ValueError(
      "Seq2seq datapoint has more live rows than the requested padded batch size."
    )
  if int(src_tokens.shape[1]) > int(padded_src_length):
    raise ValueError(
      "Seq2seq datapoint source width exceeds the requested padded source length."
    )
  if int(prev_output_tokens.shape[1]) > int(padded_tgt_length):
    raise ValueError(
      "Seq2seq datapoint target width exceeds the requested padded target length."
    )
  if live_target_count > int(padded_batch_size) * int(padded_tgt_length):
    raise ValueError(
      "Seq2seq datapoint live target count exceeds the requested padded target capacity."
    )

  padded_src_tokens = jnp.full(
    (int(padded_batch_size), int(padded_src_length)),
    int(pad_id),
    dtype=src_tokens.dtype,
  )
  padded_prev_output_tokens = jnp.full(
    (int(padded_batch_size), int(padded_tgt_length)),
    int(pad_id),
    dtype=prev_output_tokens.dtype,
  )
  padded_targets = jnp.full(
    (int(padded_batch_size) * int(padded_tgt_length),),
    int(pad_id),
    dtype=targets.dtype,
  )

  if live_batch_size > 0:
    padded_src_tokens = padded_src_tokens.at[
      :live_batch_size, -int(src_tokens.shape[1]):
    ].set(src_tokens[:live_batch_size])
    padded_prev_output_tokens = padded_prev_output_tokens.at[
      :live_batch_size, :int(prev_output_tokens.shape[1])
    ].set(prev_output_tokens[:live_batch_size])
  if live_target_count > 0:
    padded_targets = padded_targets.at[:live_target_count].set(targets[:live_target_count])

  return (padded_src_tokens, padded_prev_output_tokens), padded_targets


def describe_seq2seq_datapoint(datapoint, *, pad_id: int = SEQ2SEQ_DEFAULT_PAD_ID):
  """Return stable diagnostic metadata for a seq2seq datapoint."""
  layout = infer_seq2seq_batch_layout(datapoint, pad_id=pad_id)
  inputs, targets = datapoint
  src_tokens, prev_output_tokens = unpack_seq2seq_inputs(inputs)
  return {
    "src_tokens": [int(dim) for dim in jnp.asarray(src_tokens).shape],
    "prev_output_tokens": [int(dim) for dim in jnp.asarray(prev_output_tokens).shape],
    "targets": [int(dim) for dim in jnp.asarray(targets).shape],
    "live_batch_size": int(sum(int(length) > 0 for length in layout["target_lengths"])),
    "live_target_token_count": int(seq2seq_target_count(layout["target_lengths"])),
  }


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


def _periodic_event_due(*, step: int, total_steps: int, every: int) -> bool:
  return (int(step) % int(every) == 0) and ((int(step) != 0) or (int(total_steps) == 1))


def _resolve_optional_nonnegative_int(value: Any, *, name: str) -> int | None:
  if value is None:
    return None
  resolved = int(value)
  if resolved < 0:
    raise ValueError(f"`{name}` must be non-negative or null, got {value!r}.")
  return resolved


def translation_bleu_eval_mode(
    *,
    step: int,
    total_steps: int,
    enabled: bool,
    freq,
    test_eval_freq: int,
    full_eval_at_end: bool,
    full_eval_freq,
) -> str | None:
  """Return `sampled`, `full`, or `None` for the BLEU eval due at this step."""
  if not enabled:
    return None

  step = int(step)
  total_steps = int(total_steps)
  is_final_step = total_steps > 0 and step == total_steps - 1
  full_due = bool(full_eval_at_end) and is_final_step

  resolved_full_freq = _resolve_optional_nonnegative_int(
    full_eval_freq, name="translation_eval.full_eval_freq"
  )
  if resolved_full_freq:
    full_due = full_due or _periodic_event_due(
      step=step,
      total_steps=total_steps,
      every=resolved_full_freq,
    )
  if full_due:
    return "full"

  resolved_freq = _resolve_optional_nonnegative_int(
    freq, name="translation_eval.freq"
  )
  if resolved_freq is None:
    resolved_freq = int(test_eval_freq)
  if resolved_freq == 0:
    return None
  if _periodic_event_due(step=step, total_steps=total_steps, every=resolved_freq):
    return "sampled"
  return None


def decode_seq2seq_text(dictionary, token_ids, *, remove_bpe, detokenize: Callable[[str], str], unk_string: str) -> str:
  text = dictionary.string(token_ids, remove_bpe=remove_bpe, unk_string=unk_string)
  return detokenize(text)


def translation_next_log_probs(next_log_probs_fn, src_tokens, prev_output_tokens):
  src_tokens = jnp.asarray(src_tokens, dtype=jnp.int32)
  prev_output_tokens = jnp.asarray(prev_output_tokens, dtype=jnp.int32)
  return jnp.asarray(next_log_probs_fn(src_tokens, prev_output_tokens))


def _fairseq_max_len_from_source_width(
    src_tokens,
    *,
    max_len_a: float,
    max_len_b: int,
    max_target_positions: int,
) -> int:
  if int(max_target_positions) < 2:
    raise ValueError("max_target_positions must be at least 2 for seq2seq generation.")
  src_width = int(np.asarray(src_tokens).shape[-1])
  max_len = min(
    int(float(max_len_a) * src_width + int(max_len_b)),
    int(max_target_positions) - 1,
  )
  if _FAIRSEQ_DEFAULT_MIN_LEN > max_len:
    raise ValueError("min_len cannot be larger than max_len for seq2seq generation.")
  return int(max_len)


def _mask_fairseq_step_log_probs(
    log_probs,
    *,
    step: int,
    max_len: int,
    pad_id: int,
    eos_id: int,
):
  log_probs = np.asarray(log_probs, dtype=np.float32).copy()
  log_probs[~np.isfinite(log_probs)] = -np.inf
  log_probs[int(pad_id)] = -np.inf
  if int(step) >= int(max_len):
    eos_log_prob = log_probs[int(eos_id)]
    log_probs.fill(-np.inf)
    log_probs[int(eos_id)] = eos_log_prob
  elif int(step) < _FAIRSEQ_DEFAULT_MIN_LEN:
    log_probs[int(eos_id)] = -np.inf
  return log_probs


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
  max_len = _fairseq_max_len_from_source_width(
    src_tokens,
    max_len_a=max_len_a,
    max_len_b=max_len_b,
    max_target_positions=max_target_positions,
  )

  active_beams = [Seq2SeqBeamCandidate(tokens=(int(eos_id),), score=0.0, finished=False)]
  finalized_beams = []
  for step in range(max_len + 1):
    if not active_beams:
      break

    repeated_src = np.repeat(src_tokens[None, :], len(active_beams), axis=0)
    active_prev_output = np.asarray([beam.tokens for beam in active_beams], dtype=np.int32)
    step_log_probs = np.asarray(
      translation_next_log_probs(next_log_probs_fn, repeated_src, active_prev_output),
      dtype=np.float32,
    )
    if step_log_probs.ndim != 2 or step_log_probs.shape[0] != len(active_beams):
      raise ValueError(
        "`next_log_probs_fn` must return shape `(active_beams, vocab_size)` "
        f"for scalar beam search, got {step_log_probs.shape}."
      )

    masked_log_probs = np.stack(
      [
        _mask_fairseq_step_log_probs(
          step_log_probs[beam_index],
          step=step,
          max_len=max_len,
          pad_id=pad_id,
          eos_id=eos_id,
        )
        for beam_index in range(len(active_beams))
      ],
      axis=0,
    )
    cumulative_scores = masked_log_probs + np.asarray(
      [beam.score for beam in active_beams],
      dtype=np.float32,
    )[:, None]
    flat_scores = cumulative_scores.reshape(-1)
    top_flat_indices = np.argsort(-flat_scores)[:min(2 * int(beam_size), flat_scores.size)]

    candidates = []
    vocab_size = cumulative_scores.shape[1]
    for flat_index in top_flat_indices.tolist():
      candidate_score = float(flat_scores[flat_index])
      if not np.isfinite(candidate_score):
        continue
      beam_index, token_id = divmod(int(flat_index), int(vocab_size))
      token_id = int(token_id)
      candidates.append(
        Seq2SeqBeamCandidate(
          tokens=active_beams[beam_index].tokens + (token_id,),
          score=candidate_score,
          finished=bool(token_id == int(eos_id)),
        )
      )

    for candidate in candidates[:int(beam_size)]:
      if candidate.finished and len(finalized_beams) < int(beam_size):
        finalized_beams.append(
          Seq2SeqBeamCandidate(
            tokens=candidate.tokens,
            score=float(
              candidate.score / ((step + 1) ** _FAIRSEQ_DEFAULT_LEN_PENALTY)
            ),
            finished=True,
          )
        )

    if len(finalized_beams) == int(beam_size) or step == max_len:
      break

    active_beams = [
      Seq2SeqBeamCandidate(
        tokens=candidate.tokens,
        score=candidate.score,
        finished=False,
      )
      for candidate in candidates
      if not candidate.finished
    ][:int(beam_size)]

  best = max(finalized_beams or active_beams, key=lambda candidate: candidate.score)
  return np.asarray(best.tokens[1:], dtype=np.int32)


def generate_seq2seq_batched_beam_search(
    *,
    next_log_probs_fn,
    src_tokens,
    beam_size: int,
    max_len_a: float,
    max_len_b: int,
    pad_id: int,
    eos_id: int,
    max_target_positions: int,
) -> list[np.ndarray]:
  """Batched beam search that issues one model call per generation step."""
  if beam_size <= 0:
    raise ValueError(f"beam_size must be positive, got {beam_size}.")

  src_tokens = np.asarray(src_tokens, dtype=np.int32)
  if src_tokens.ndim != 2:
    raise ValueError(f"`src_tokens` must be rank-2, got shape {src_tokens.shape}.")
  if src_tokens.shape[0] == 0:
    return []

  max_generated_lengths = []
  for row in src_tokens:
    src_length = seq2seq_source_length(row, pad_id=pad_id, eos_id=eos_id)
    max_generated_length = int(max_len_a * src_length + max_len_b)
    max_generated_length = max(1, min(int(max_target_positions) - 1, max_generated_length))
    max_generated_lengths.append(max_generated_length)
  if min(max_generated_lengths) <= 0:
    raise ValueError("max_target_positions must be at least 2 for seq2seq generation.")
  max_generated_length = max(max_generated_lengths)
  max_active_count = src_tokens.shape[0] * int(beam_size)
  fixed_prev_output_length = max_generated_length + 1

  beams_by_example = [
    [Seq2SeqBeamCandidate(tokens=(int(eos_id),), score=0.0, finished=False)]
    for _ in range(src_tokens.shape[0])
  ]

  for step in range(max_generated_length):
    active_records = []
    for example_index, beams in enumerate(beams_by_example):
      if step >= max_generated_lengths[example_index]:
        continue
      for beam in beams:
        if not beam.finished:
          active_records.append((example_index, beam))
    if not active_records:
      break

    active_count = len(active_records)
    active_src = np.empty(
      (max_active_count, src_tokens.shape[1]),
      dtype=np.int32,
    )
    active_prev_output = np.full(
      (max_active_count, fixed_prev_output_length),
      int(pad_id),
      dtype=np.int32,
    )
    for record_index, (example_index, beam) in enumerate(active_records):
      active_src[record_index] = src_tokens[example_index]
      active_prev_output[record_index, :len(beam.tokens)] = beam.tokens
    if active_count < max_active_count:
      active_src[active_count:] = src_tokens[0]
      active_prev_output[active_count:, 0] = int(eos_id)
    step_log_probs = np.asarray(
      translation_next_log_probs(next_log_probs_fn, active_src, active_prev_output),
      dtype=np.float32,
    )[:active_count]

    candidates_by_example = [
      [beam for beam in beams if beam.finished]
      for beams in beams_by_example
    ]
    for record_index, (example_index, beam) in enumerate(active_records):
      top_indices = np.argsort(-step_log_probs[record_index])[:beam_size]
      for token_id in top_indices.tolist():
        token_id = int(token_id)
        candidates_by_example[example_index].append(
          Seq2SeqBeamCandidate(
            tokens=beam.tokens + (token_id,),
            score=float(beam.score + step_log_probs[record_index, token_id]),
            finished=bool(token_id == int(eos_id)),
          )
        )

    for example_index, candidates in enumerate(candidates_by_example):
      if not candidates:
        candidates = beams_by_example[example_index]
      candidates.sort(key=lambda candidate: candidate.score, reverse=True)
      beams_by_example[example_index] = candidates[:beam_size]

    if all(all(beam.finished for beam in beams) for beams in beams_by_example):
      break

  best_tokens = []
  for beams in beams_by_example:
    completed = [beam for beam in beams if beam.finished]
    best = max(completed or beams, key=lambda candidate: candidate.score)
    best_tokens.append(np.asarray(best.tokens[1:], dtype=np.int32))
  return best_tokens


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
    max_examples: int | None = None,
    progress_freq: int | None = None,
    progress_callback: Callable[[int], None] | None = None,
    use_batched_generation: bool = True,
):
  if hasattr(dataloader, "reset"):
    dataloader.reset()
  if max_examples is not None and int(max_examples) <= 0:
    return {
      "bleu": 0.0,
      "hypotheses": [],
      "references": [],
      "sample_hypothesis": None,
      "sample_reference": None,
      "num_examples": 0,
    }

  detokenize = build_translation_detokenizer(detok, target_lang=target_lang)
  hypotheses = []
  references = []
  sample_hypothesis = None
  sample_reference = None

  for x_batch, y_batch in dataloader:
    src_tokens, prev_output_tokens = unpack_seq2seq_inputs(x_batch)
    target_sequences = recover_seq2seq_target_sequences(prev_output_tokens, y_batch, pad_id=pad_id)
    src_tokens = np.asarray(src_tokens, dtype=np.int32)
    live_examples = []
    for example_index, reference_tokens in enumerate(target_sequences):
      if (
          max_examples is not None
          and len(hypotheses) + len(live_examples) >= int(max_examples)
      ):
        break
      if len(reference_tokens) == 0:
        continue
      live_examples.append((example_index, reference_tokens))
    if not live_examples:
      continue

    if use_batched_generation:
      batch_hypothesis_tokens = generate_seq2seq_batched_beam_search(
        next_log_probs_fn=next_log_probs_fn,
        src_tokens=np.asarray(
          [src_tokens[example_index] for example_index, _ in live_examples],
          dtype=np.int32,
        ),
        beam_size=beam_size,
        max_len_a=max_len_a,
        max_len_b=max_len_b,
        pad_id=pad_id,
        eos_id=eos_id,
        max_target_positions=max_target_positions,
      )
    else:
      batch_hypothesis_tokens = [
        generate_seq2seq_beam_search(
          next_log_probs_fn=next_log_probs_fn,
          src_tokens=src_tokens[example_index],
          beam_size=beam_size,
          max_len_a=max_len_a,
          max_len_b=max_len_b,
          pad_id=pad_id,
          eos_id=eos_id,
          max_target_positions=max_target_positions,
        )
        for example_index, _ in live_examples
      ]

    for hypothesis_tokens, (_, reference_tokens) in zip(batch_hypothesis_tokens, live_examples):
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
      if (
          progress_callback is not None
          and progress_freq is not None
          and int(progress_freq) > 0
          and len(hypotheses) % int(progress_freq) == 0
      ):
        progress_callback(len(hypotheses))
    if max_examples is not None and len(hypotheses) >= int(max_examples):
      break

  if not hypotheses:
    return {
      "bleu": 0.0,
      "hypotheses": hypotheses,
      "references": references,
      "sample_hypothesis": sample_hypothesis,
      "sample_reference": sample_reference,
      "num_examples": 0,
    }

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
