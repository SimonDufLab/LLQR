import math
from typing import NamedTuple, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from lqr_optimizer._src.utils.seq2seq_utils import unpack_seq2seq_inputs
from lqr_optimizer._src.utils.utils import (
  EnhancedSequential,
  make_controlled_stage_descriptor,
  make_lqr_segment_descriptor,
  make_passive_stage_descriptor,
)


def make_padding_mask(tokens: jnp.ndarray, pad_id: int) -> jnp.ndarray:
  return (tokens != pad_id).astype(jnp.float32)[:, None, None, :]


def make_causal_mask(seq_len: int, batch_size: int) -> jnp.ndarray:
  mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))
  return jnp.broadcast_to(mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))


def combine_masks(*masks: jnp.ndarray) -> jnp.ndarray:
  result = masks[0]
  for mask in masks[1:]:
    result = result * mask
  return result


def _embedding_init(*, embedding_dim: int, pad_id: int):
  def init(key, shape, dtype=jnp.float32):
    embedding = nn.initializers.normal(stddev=embedding_dim ** -0.5)(key, shape, dtype)
    return embedding.at[pad_id].set(jnp.zeros((shape[1],), dtype=dtype))

  return init


def _sinusoidal_position_encoding(seq_len: int, dim: int, dtype) -> jnp.ndarray:
  positions = jnp.arange(seq_len, dtype=dtype)[:, None]
  div_term = jnp.exp(jnp.arange(0, dim, 2, dtype=dtype) * (-math.log(10000.0) / dim))
  encoding = jnp.zeros((seq_len, dim), dtype=dtype)
  encoding = encoding.at[:, 0::2].set(jnp.sin(positions * div_term))
  encoding = encoding.at[:, 1::2].set(jnp.cos(positions * div_term))
  return encoding[None, :, :]


def _validate_tokens(tokens: jnp.ndarray, *, expected_rank: int, max_positions: int, field_name: str) -> None:
  if tokens.ndim != expected_rank:
    raise ValueError(f"`{field_name}` must be rank-{expected_rank}, got shape {tokens.shape}.")
  if not jnp.issubdtype(tokens.dtype, jnp.integer):
    raise ValueError(f"`{field_name}` must contain integer token ids, got dtype {tokens.dtype}.")
  if tokens.shape[1] > max_positions:
    raise ValueError(
      f"`{field_name}` length {tokens.shape[1]} exceeds max_positions={max_positions}."
    )


class TranslationState(NamedTuple):
  src_h: jnp.ndarray
  tgt_h: jnp.ndarray
  tgt_embed_matrix: jnp.ndarray
  tgt_nonpad_mask: jnp.ndarray
  src_pad_mask: jnp.ndarray
  tgt_self_mask: jnp.ndarray
  src_residual: Optional[jnp.ndarray]
  tgt_residual: Optional[jnp.ndarray]


class TranslationInitStage(nn.Module):
  src_embedding: nn.Module
  tgt_embedding: nn.Module
  emb_dim: int
  dropout_rate: float
  max_source_positions: int
  max_target_positions: int
  pad_id: int
  deterministic: bool

  @nn.compact
  def __call__(self, x):
    src_tokens, prev_output_tokens = unpack_seq2seq_inputs(x)
    _validate_tokens(
      src_tokens,
      expected_rank=2,
      max_positions=self.max_source_positions,
      field_name="src_tokens",
    )
    _validate_tokens(
      prev_output_tokens,
      expected_rank=2,
      max_positions=self.max_target_positions,
      field_name="prev_output_tokens",
    )

    embed_scale = math.sqrt(self.emb_dim)
    src_h = embed_scale * self.src_embedding(src_tokens)
    tgt_h = embed_scale * self.tgt_embedding(prev_output_tokens)
    tgt_embed_matrix = self.tgt_embedding.embedding

    src_h = src_h + _sinusoidal_position_encoding(src_tokens.shape[1], self.emb_dim, src_h.dtype)
    tgt_h = tgt_h + _sinusoidal_position_encoding(prev_output_tokens.shape[1], self.emb_dim, tgt_h.dtype)

    src_h = nn.Dropout(rate=self.dropout_rate)(src_h, deterministic=self.deterministic)
    tgt_h = nn.Dropout(rate=self.dropout_rate)(tgt_h, deterministic=self.deterministic)

    tgt_nonpad_mask = (prev_output_tokens != self.pad_id).astype(jnp.float32)
    src_pad_mask = make_padding_mask(src_tokens, self.pad_id)
    tgt_self_mask = combine_masks(
      make_padding_mask(prev_output_tokens, self.pad_id),
      make_causal_mask(prev_output_tokens.shape[1], prev_output_tokens.shape[0]),
    )
    return TranslationState(
      src_h,
      tgt_h,
      tgt_embed_matrix,
      tgt_nonpad_mask,
      src_pad_mask,
      tgt_self_mask,
      None,
      None,
    )


class EncoderSelfAttentionStage(nn.Module):
  emb_dim: int
  num_heads: int
  attention_dropout_rate: float = 0.0
  deterministic: bool = True

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    residual = state.src_h
    attended = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      qkv_features=self.emb_dim,
      out_features=self.emb_dim,
      dropout_rate=self.attention_dropout_rate,
      broadcast_dropout=False,
      deterministic=self.deterministic,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.zeros,
      name="MultiHeadDotProductAttention_0",
    )(residual, residual, mask=state.src_pad_mask > 0)
    return state._replace(src_h=attended, src_residual=residual)


class DecoderSelfAttentionStage(nn.Module):
  emb_dim: int
  num_heads: int
  attention_dropout_rate: float = 0.0
  deterministic: bool = True

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    residual = state.tgt_h
    attended = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      qkv_features=self.emb_dim,
      out_features=self.emb_dim,
      dropout_rate=self.attention_dropout_rate,
      broadcast_dropout=False,
      deterministic=self.deterministic,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.zeros,
      name="MultiHeadDotProductAttention_0",
    )(residual, residual, mask=state.tgt_self_mask > 0)
    return state._replace(tgt_h=attended, tgt_residual=residual)


class DecoderCrossAttentionStage(nn.Module):
  emb_dim: int
  num_heads: int
  attention_dropout_rate: float = 0.0
  deterministic: bool = True

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    residual = state.tgt_h
    attended = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      qkv_features=self.emb_dim,
      out_features=self.emb_dim,
      dropout_rate=self.attention_dropout_rate,
      broadcast_dropout=False,
      deterministic=self.deterministic,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.zeros,
      name="MultiHeadDotProductAttention_0",
    )(residual, state.src_h, mask=state.src_pad_mask > 0)
    return state._replace(tgt_h=attended, tgt_residual=residual)


class PositionwiseFeedForwardStage(nn.Module):
  mlp_dim: int
  carry_source: str

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    if self.carry_source == "src":
      residual = state.src_h
      projected = nn.Dense(
        self.mlp_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        name="Dense_0",
      )(state.src_h)
      return state._replace(src_h=projected, src_residual=residual)

    if self.carry_source == "tgt":
      residual = state.tgt_h
      projected = nn.Dense(
        self.mlp_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        name="Dense_0",
      )(state.tgt_h)
      return state._replace(tgt_h=projected, tgt_residual=residual)

    raise ValueError(f"Unsupported carry_source '{self.carry_source}'.")


class PositionwiseReluStage(nn.Module):
  carry_source: str
  activation_dropout_rate: float = 0.0
  deterministic: bool = True

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    if self.carry_source == "src":
      activated = nn.relu(state.src_h)
      activated = nn.Dropout(rate=self.activation_dropout_rate)(activated, deterministic=self.deterministic)
      return state._replace(src_h=activated)

    if self.carry_source == "tgt":
      activated = nn.relu(state.tgt_h)
      activated = nn.Dropout(rate=self.activation_dropout_rate)(activated, deterministic=self.deterministic)
      return state._replace(tgt_h=activated)

    raise ValueError(f"Unsupported carry_source '{self.carry_source}'.")


class EncoderResidualNormStage(nn.Module):
  dropout_rate: float
  output_dim: Optional[int] = None
  apply_output_projection: bool = False
  layer_norm_eps: float = 1e-5
  layer_norm_name: str = "LayerNorm_0"
  deterministic: bool = True

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    if state.src_residual is None:
      raise ValueError("Encoder residual stage requires a source residual.")

    x = state.src_h
    if self.apply_output_projection:
      if self.output_dim is None:
        raise ValueError("output_dim must be provided when apply_output_projection=True.")
      x = nn.Dense(
        self.output_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        name="Dense_0",
      )(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    normalized = nn.LayerNorm(epsilon=self.layer_norm_eps, name=self.layer_norm_name)(state.src_residual + x)
    return state._replace(src_h=normalized, src_residual=None)


class DecoderResidualNormStage(nn.Module):
  dropout_rate: float
  output_dim: Optional[int] = None
  apply_output_projection: bool = False
  layer_norm_eps: float = 1e-5
  layer_norm_name: str = "LayerNorm_0"
  deterministic: bool = True

  @nn.compact
  def __call__(self, state: TranslationState) -> TranslationState:
    if state.tgt_residual is None:
      raise ValueError("Decoder residual stage requires a target residual.")

    x = state.tgt_h
    if self.apply_output_projection:
      if self.output_dim is None:
        raise ValueError("output_dim must be provided when apply_output_projection=True.")
      x = nn.Dense(
        self.output_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros,
        name="Dense_0",
      )(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=self.deterministic)
    normalized = nn.LayerNorm(epsilon=self.layer_norm_eps, name=self.layer_norm_name)(state.tgt_residual + x)
    return state._replace(tgt_h=normalized, tgt_residual=None)


class TranslationFinalLogitsStage(nn.Module):
  def __call__(self, state: TranslationState) -> jnp.ndarray:
    logits = jnp.einsum("btd,vd->btv", state.tgt_h, state.tgt_embed_matrix)
    batch_size, target_length, vocab_size = logits.shape
    flat_logits = logits.reshape((batch_size * target_length, vocab_size))
    flat_valid = (state.tgt_nonpad_mask.reshape((batch_size * target_length,)) > 0).astype(jnp.int32)
    flat_indices = jnp.arange(batch_size * target_length, dtype=jnp.int32)
    sort_keys = (1 - flat_valid) * (batch_size * target_length) + flat_indices
    sort_order = jnp.argsort(sort_keys, stable=True)
    return flat_logits[sort_order]


class TranslationFinalLogSoftmaxStage(nn.Module):
  def __call__(self, logits: jnp.ndarray) -> jnp.ndarray:
    return nn.log_softmax(logits, axis=-1)


class TranslationEnhancedSequential(EnhancedSequential):
  pad_id: int = 1
  bos_id: int = 0
  eos_id: int = 2
  max_source_positions: int = 1024
  max_target_positions: int = 1024

  def _translation_state_before_readout(self, x) -> TranslationState:
    state = x
    for block, stage in zip(self.layers, self.execution_stage_descriptors):
      if stage.name == "translation_logits":
        break
      state = block(state)
    if not isinstance(state, TranslationState):
      raise ValueError(
        "Translation decode helpers expected a TranslationState before readout."
      )
    return state

  def next_log_probs(self, x) -> jnp.ndarray:
    """Return log-probabilities for the final non-pad decoder position."""
    state = self._translation_state_before_readout(x)
    batch_size = state.tgt_h.shape[0]
    target_lengths = jnp.sum(state.tgt_nonpad_mask > 0, axis=1).astype(jnp.int32)
    last_indices = jnp.maximum(target_lengths - 1, 0)
    last_hidden = state.tgt_h[jnp.arange(batch_size), last_indices]
    logits = jnp.einsum("bd,vd->bv", last_hidden, state.tgt_embed_matrix)
    return nn.log_softmax(logits, axis=-1)


def _append_controlled(layers, stage_descriptors, stage_name, module):
  layer_index = len(layers)
  layers.append(module)
  stage_descriptors.append(
    make_controlled_stage_descriptor(stage_name, f"layers_{layer_index}")
  )


def _append_passive(layers, stage_descriptors, stage_name, module, *, fast_path_kind=None,
                    passive_state_hessian=None):
  layers.append(module)
  stage_descriptors.append(
    make_passive_stage_descriptor(
      stage_name,
      fast_path_kind=fast_path_kind,
      passive_state_hessian=passive_state_hessian,
    )
  )


def _append_encoder_block(layers, stage_descriptors, lqr_segment_descriptors, *, block_index, emb_dim,
                          width_mlp, num_heads, dropout, attention_dropout, activation_dropout,
                          layer_norm_eps, deterministic):
  stage_names = (
    f"encoder_{block_index}_self_attn",
    f"encoder_{block_index}_self_attn_residual_norm",
    f"encoder_{block_index}_fc1",
    f"encoder_{block_index}_relu",
    f"encoder_{block_index}_fc2_residual_norm",
  )

  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[0],
    EncoderSelfAttentionStage(
      emb_dim=emb_dim,
      num_heads=num_heads,
      attention_dropout_rate=attention_dropout,
      deterministic=deterministic,
    ),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[1],
    EncoderResidualNormStage(
      dropout_rate=dropout,
      layer_norm_eps=layer_norm_eps,
      deterministic=deterministic,
    ),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[2],
    PositionwiseFeedForwardStage(
      mlp_dim=width_mlp,
      carry_source="src",
    ),
  )
  _append_passive(
    layers,
    stage_descriptors,
    stage_names[3],
    PositionwiseReluStage(
      carry_source="src",
      activation_dropout_rate=activation_dropout,
      deterministic=deterministic,
    ),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[4],
    EncoderResidualNormStage(
      dropout_rate=dropout,
      output_dim=emb_dim,
      apply_output_projection=True,
      layer_norm_eps=layer_norm_eps,
      deterministic=deterministic,
    ),
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      f"encoder_block_{block_index}",
      stage_names,
      sample_separable_second_order=bool(deterministic),
    )
  )


def _append_decoder_block(layers, stage_descriptors, lqr_segment_descriptors, *, block_index, emb_dim,
                          width_mlp, num_heads, dropout, attention_dropout, activation_dropout,
                          layer_norm_eps, deterministic):
  stage_names = (
    f"decoder_{block_index}_self_attn",
    f"decoder_{block_index}_self_attn_residual_norm",
    f"decoder_{block_index}_cross_attn",
    f"decoder_{block_index}_cross_attn_residual_norm",
    f"decoder_{block_index}_fc1",
    f"decoder_{block_index}_relu",
    f"decoder_{block_index}_fc2_residual_norm",
  )

  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[0],
    DecoderSelfAttentionStage(
      emb_dim=emb_dim,
      num_heads=num_heads,
      attention_dropout_rate=attention_dropout,
      deterministic=deterministic,
    ),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[1],
    DecoderResidualNormStage(
      dropout_rate=dropout,
      layer_norm_eps=layer_norm_eps,
      deterministic=deterministic,
    ),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[2],
    DecoderCrossAttentionStage(
      emb_dim=emb_dim,
      num_heads=num_heads,
      attention_dropout_rate=attention_dropout,
      deterministic=deterministic,
    ),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[3],
    DecoderResidualNormStage(
      dropout_rate=dropout,
      layer_norm_eps=layer_norm_eps,
      deterministic=deterministic,
    ),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[4],
    PositionwiseFeedForwardStage(
      mlp_dim=width_mlp,
      carry_source="tgt",
    ),
  )
  _append_passive(
    layers,
    stage_descriptors,
    stage_names[5],
    PositionwiseReluStage(
      carry_source="tgt",
      activation_dropout_rate=activation_dropout,
      deterministic=deterministic,
    ),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[6],
    DecoderResidualNormStage(
      dropout_rate=dropout,
      output_dim=emb_dim,
      apply_output_projection=True,
      layer_norm_eps=layer_norm_eps,
      deterministic=deterministic,
    ),
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      f"decoder_block_{block_index}",
      stage_names,
      sample_separable_second_order=bool(deterministic),
    )
  )


def create_transformer_iwslt_model(
    num_classes: int,
    *,
    src_vocab_size: int,
    tgt_vocab_size: Optional[int] = None,
    emb_dim: int = 512,
    num_heads: int = 4,
    encoder_depth: int = 6,
    decoder_depth: int = 6,
    width_mlp: int = 1024,
    dropout: float = 0.3,
    attention_dropout: float = 0.0,
    activation_dropout: float = 0.0,
    layer_norm_eps: float = 1e-5,
    max_source_positions: int = 1024,
    max_target_positions: int = 1024,
    pad_id: int = 1,
    bos_id: int = 0,
    eos_id: int = 2,
    tie_output_to_target_embedding: bool = True,
) -> Tuple[EnhancedSequential, EnhancedSequential]:
  if tgt_vocab_size is None:
    tgt_vocab_size = num_classes
  if int(tgt_vocab_size) != int(num_classes):
    raise ValueError(
      f"tgt_vocab_size ({tgt_vocab_size}) must match num_classes ({num_classes})."
    )
  if emb_dim % num_heads != 0:
    raise ValueError("emb_dim must be divisible by num_heads.")
  if not tie_output_to_target_embedding:
    raise ValueError("Wave 3 requires tie_output_to_target_embedding=True.")
  if encoder_depth != 6 or decoder_depth != 6:
    # Allowed for tests and bounded smokes; keep public metadata contract on defaults.
    pass

  def build_model(*, deterministic: bool) -> EnhancedSequential:
    src_embedding = nn.Embed(
      num_embeddings=src_vocab_size,
      features=emb_dim,
      embedding_init=_embedding_init(embedding_dim=emb_dim, pad_id=pad_id),
      name="src_embed",
    )
    tgt_embedding = nn.Embed(
      num_embeddings=tgt_vocab_size,
      features=emb_dim,
      embedding_init=_embedding_init(embedding_dim=emb_dim, pad_id=pad_id),
      name="tgt_embed",
    )

    layers = []
    stage_descriptors = []
    lqr_segment_descriptors = []

    _append_controlled(
      layers,
      stage_descriptors,
      "translation_init",
      TranslationInitStage(
        src_embedding=src_embedding,
        tgt_embedding=tgt_embedding,
        emb_dim=emb_dim,
        dropout_rate=dropout,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        pad_id=pad_id,
        deterministic=deterministic,
      ),
    )
    lqr_segment_descriptors.append(
      make_lqr_segment_descriptor(
        "translation_init",
        ("translation_init",),
        sample_separable_second_order=bool(deterministic),
      )
    )

    for block_index in range(encoder_depth):
      _append_encoder_block(
        layers,
        stage_descriptors,
        lqr_segment_descriptors,
        block_index=block_index,
        emb_dim=emb_dim,
        width_mlp=width_mlp,
        num_heads=num_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        layer_norm_eps=layer_norm_eps,
        deterministic=deterministic,
      )

    for block_index in range(decoder_depth):
      _append_decoder_block(
        layers,
        stage_descriptors,
        lqr_segment_descriptors,
        block_index=block_index,
        emb_dim=emb_dim,
        width_mlp=width_mlp,
        num_heads=num_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        layer_norm_eps=layer_norm_eps,
        deterministic=deterministic,
      )

    _append_passive(
      layers,
      stage_descriptors,
      "translation_logits",
      TranslationFinalLogitsStage(),
      passive_state_hessian="zero",
    )
    _append_passive(
      layers,
      stage_descriptors,
      "translation_log_softmax",
      TranslationFinalLogSoftmaxStage(),
      passive_state_hessian="generic",
    )
    lqr_segment_descriptors.append(
      make_lqr_segment_descriptor(
        "translation_final",
        ("translation_logits", "translation_log_softmax"),
        sample_separable_second_order=bool(deterministic),
      )
    )

    model = TranslationEnhancedSequential(
      layers,
      stage_descriptors=tuple(stage_descriptors),
      lqr_segment_descriptors=tuple(lqr_segment_descriptors),
      pad_id=pad_id,
      bos_id=bos_id,
      eos_id=eos_id,
      max_source_positions=max_source_positions,
      max_target_positions=max_target_positions,
    )
    model.validate_stage_descriptors()
    model.validate_lqr_segment_descriptors()
    return model

  return build_model(deterministic=False), build_model(deterministic=True)
