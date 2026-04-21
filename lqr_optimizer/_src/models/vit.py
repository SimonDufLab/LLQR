from typing import Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp

from lqr_optimizer._src.utils.utils import (
  EnhancedSequential,
  make_controlled_stage_descriptor,
  make_lqr_segment_descriptor,
  make_passive_stage_descriptor,
)


_VIT_NUM_LAYERS = 12
_VIT_DROPOUT_RATE = 0.1
_VIT_ATTENTION_DROPOUT_RATE = 0.1
_MLP_KERNEL_INIT = nn.initializers.xavier_uniform()
_MLP_BIAS_INIT = nn.initializers.normal(stddev=1e-6)
_POSITION_EMBEDDING_INIT = nn.initializers.normal(stddev=0.02)


class PatchEmbeddingStage(nn.Module):
  hidden_size: int
  patch_size: Tuple[int, int]
  conv_name: str = "Conv_0"

  @nn.compact
  def __call__(self, x):
    return nn.Conv(
      features=self.hidden_size,
      kernel_size=self.patch_size,
      strides=self.patch_size,
      padding="VALID",
      name=self.conv_name,
    )(x)


class PatchFlattenStage(nn.Module):
  def __call__(self, x):
    batch_size, height, width, channels = x.shape
    return jnp.reshape(x, (batch_size, height * width, channels))


class AddClassTokenStage(nn.Module):
  @nn.compact
  def __call__(self, x):
    cls_token = self.param("cls", nn.initializers.zeros, (1, 1, x.shape[-1]))
    cls_token = jnp.tile(cls_token, (x.shape[0], 1, 1))
    return jnp.concatenate((cls_token, x), axis=1)


class AddPositionEmbeddingStage(nn.Module):
  @nn.compact
  def __call__(self, x):
    position_embedding = self.param(
      "pos_embedding",
      _POSITION_EMBEDDING_INIT,
      (1, x.shape[1], x.shape[2]),
    )
    return x + position_embedding


class EmbeddingDropoutStage(nn.Module):
  rate: float = _VIT_DROPOUT_RATE
  deterministic: bool = True

  @nn.compact
  def __call__(self, x):
    return nn.Dropout(rate=self.rate)(x, deterministic=self.deterministic)


class LayerNormCarryStage(nn.Module):
  layer_norm_name: str = "LayerNorm_0"

  @nn.compact
  def __call__(self, x):
    normalized = nn.LayerNorm(name=self.layer_norm_name)(x)
    return normalized, x


class AttentionCoreFromCarryStage(nn.Module):
  hidden_size: int
  num_heads: int
  attention_dropout_rate: float = _VIT_ATTENTION_DROPOUT_RATE
  deterministic: bool = True

  @nn.compact
  def __call__(self, inputs):
    x, residual = inputs
    attended = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads,
      dropout_rate=self.attention_dropout_rate,
      broadcast_dropout=False,
      deterministic=self.deterministic,
      kernel_init=nn.initializers.xavier_uniform(),
      name="MultiHeadDotProductAttention_0",
    )(x, x)
    return attended, residual


class ResidualAddDropoutStage(nn.Module):
  rate: float = _VIT_DROPOUT_RATE
  deterministic: bool = True

  @nn.compact
  def __call__(self, inputs):
    x, residual = inputs
    x = nn.Dropout(rate=self.rate)(x, deterministic=self.deterministic)
    return residual + x


class FC1FromCarryStage(nn.Module):
  mlp_dim: int

  @nn.compact
  def __call__(self, inputs):
    x, residual = inputs
    projected = nn.Dense(
      features=self.mlp_dim,
      kernel_init=_MLP_KERNEL_INIT,
      bias_init=_MLP_BIAS_INIT,
      name="Dense_0",
    )(x)
    return projected, residual


class GELUFromCarryStage(nn.Module):
  def __call__(self, inputs):
    x, residual = inputs
    return nn.gelu(x), residual


class MlpHiddenDropoutStage(nn.Module):
  rate: float = _VIT_DROPOUT_RATE
  deterministic: bool = True

  @nn.compact
  def __call__(self, inputs):
    x, residual = inputs
    x = nn.Dropout(rate=self.rate)(x, deterministic=self.deterministic)
    return x, residual


class FC2FromCarryStage(nn.Module):
  hidden_size: int

  @nn.compact
  def __call__(self, inputs):
    x, residual = inputs
    projected = nn.Dense(
      features=self.hidden_size,
      kernel_init=_MLP_KERNEL_INIT,
      bias_init=_MLP_BIAS_INIT,
      name="Dense_0",
    )(x)
    return projected, residual


class EncoderNormStage(nn.Module):
  layer_norm_name: str = "LayerNorm_0"

  @nn.compact
  def __call__(self, x):
    return nn.LayerNorm(name=self.layer_norm_name)(x)


class CLSReadoutStage(nn.Module):
  def __call__(self, x):
    return x[:, 0]


class IdentityStage(nn.Module):
  def __call__(self, x):
    return x


class LogSoftmaxStage(nn.Module):
  def __call__(self, x):
    return nn.log_softmax(x)


def _append_controlled(layers, stage_descriptors, stage_name, module, *, fast_path_kind=None):
  layer_index = len(layers)
  layers.append(module)
  stage_descriptors.append(
    make_controlled_stage_descriptor(stage_name, f"layers_{layer_index}", fast_path_kind=fast_path_kind)
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


def _resolve_patch_size(num_classes: int, patch_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
  if patch_size is not None:
    return patch_size
  if num_classes in (10, 100):
    return (4, 4)
  if num_classes == 1000:
    return (16, 16)
  raise ValueError("patch_size must be provided when num_classes is not 10, 100, or 1000.")


def _append_encoder_block(layers, stage_descriptors, lqr_segment_descriptors, *, block_index, hidden_size,
                          mlp_dim, num_heads, dropout_rate, attention_dropout_rate, inference):
  stage_names = (
    f"block_{block_index}_attn_pre_ln",
    f"block_{block_index}_attn_core",
    f"block_{block_index}_attn_residual",
    f"block_{block_index}_mlp_pre_ln",
    f"block_{block_index}_fc1",
    f"block_{block_index}_gelu",
    f"block_{block_index}_mlp_hidden_dropout",
    f"block_{block_index}_fc2",
    f"block_{block_index}_mlp_residual",
  )

  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[0],
    LayerNormCarryStage(),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[1],
    AttentionCoreFromCarryStage(
      hidden_size=hidden_size,
      num_heads=num_heads,
      attention_dropout_rate=attention_dropout_rate,
      deterministic=inference,
    ),
  )
  _append_passive(
    layers,
    stage_descriptors,
    stage_names[2],
    ResidualAddDropoutStage(rate=dropout_rate, deterministic=inference),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[3],
    LayerNormCarryStage(),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[4],
    FC1FromCarryStage(mlp_dim=mlp_dim),
    fast_path_kind="linear_controlled",
  )
  _append_passive(
    layers,
    stage_descriptors,
    stage_names[5],
    GELUFromCarryStage(),
  )
  _append_passive(
    layers,
    stage_descriptors,
    stage_names[6],
    MlpHiddenDropoutStage(rate=dropout_rate, deterministic=inference),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stage_names[7],
    FC2FromCarryStage(hidden_size=hidden_size),
    fast_path_kind="linear_controlled",
  )
  _append_passive(
    layers,
    stage_descriptors,
    stage_names[8],
    ResidualAddDropoutStage(rate=dropout_rate, deterministic=inference),
  )

  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      f"encoder_block_{block_index}",
      stage_names,
      sample_separable_second_order=bool(inference),
    )
  )


def _build_vit(*, num_classes: int, hidden_size: int, mlp_dim: int, num_heads: int,
               patch_size: Optional[Tuple[int, int]], num_layers: int, dropout_rate: float,
               attention_dropout_rate: float, inference: bool) -> EnhancedSequential:
  resolved_patch_size = _resolve_patch_size(num_classes, patch_size)
  layers = []
  stage_descriptors = []
  lqr_segment_descriptors = []

  stem_stage_names = (
    "stem_patch_embed",
    "stem_patch_flatten",
    "stem_cls_token",
    "stem_pos_embed",
    "stem_dropout",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stem_stage_names[0],
    PatchEmbeddingStage(hidden_size=hidden_size, patch_size=resolved_patch_size),
    fast_path_kind="linear_controlled",
  )
  _append_passive(
    layers,
    stage_descriptors,
    stem_stage_names[1],
    PatchFlattenStage(),
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stem_stage_names[2],
    AddClassTokenStage(),
  )
  _append_controlled(
    layers,
    stage_descriptors,
    stem_stage_names[3],
    AddPositionEmbeddingStage(),
  )
  _append_passive(
    layers,
    stage_descriptors,
    stem_stage_names[4],
    EmbeddingDropoutStage(rate=dropout_rate, deterministic=inference),
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      "stem",
      stem_stage_names,
      sample_separable_second_order=bool(inference),
    )
  )

  for block_index in range(num_layers):
    _append_encoder_block(
      layers,
      stage_descriptors,
      lqr_segment_descriptors,
      block_index=block_index,
      hidden_size=hidden_size,
      mlp_dim=mlp_dim,
      num_heads=num_heads,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate,
      inference=inference,
    )

  head_stage_names = (
    "encoder_norm",
    "cls_readout",
    "pre_logits",
    "logits",
    "log_softmax",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    head_stage_names[0],
    EncoderNormStage(),
  )
  _append_passive(
    layers,
    stage_descriptors,
    head_stage_names[1],
    CLSReadoutStage(),
    passive_state_hessian="zero",
  )
  _append_passive(
    layers,
    stage_descriptors,
    head_stage_names[2],
    IdentityStage(),
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    head_stage_names[3],
    nn.Dense(features=num_classes, kernel_init=nn.initializers.zeros),
    fast_path_kind="linear_controlled",
  )
  _append_passive(
    layers,
    stage_descriptors,
    head_stage_names[4],
    LogSoftmaxStage(),
    passive_state_hessian="generic",
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      "head",
      head_stage_names,
      sample_separable_second_order=bool(inference),
    )
  )

  model = EnhancedSequential(
    layers,
    stage_descriptors=tuple(stage_descriptors),
    lqr_segment_descriptors=tuple(lqr_segment_descriptors),
  )
  model.validate_stage_descriptors()
  return model


def create_vit_ti16(num_classes: int, *, patch_size: Optional[Tuple[int, int]] = None,
                    dropout_rate: float = _VIT_DROPOUT_RATE,
                    attention_dropout_rate: float = _VIT_ATTENTION_DROPOUT_RATE,
                    num_layers: int = _VIT_NUM_LAYERS) -> Tuple[EnhancedSequential, EnhancedSequential]:
  return (
    _build_vit(
      num_classes=num_classes,
      hidden_size=192,
      mlp_dim=768,
      num_heads=3,
      patch_size=patch_size,
      num_layers=num_layers,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate,
      inference=False,
    ),
    _build_vit(
      num_classes=num_classes,
      hidden_size=192,
      mlp_dim=768,
      num_heads=3,
      patch_size=patch_size,
      num_layers=num_layers,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate,
      inference=True,
    ),
  )


def create_vit_s16(num_classes: int, *, patch_size: Optional[Tuple[int, int]] = None,
                   dropout_rate: float = _VIT_DROPOUT_RATE,
                   attention_dropout_rate: float = _VIT_ATTENTION_DROPOUT_RATE,
                   num_layers: int = _VIT_NUM_LAYERS) -> Tuple[EnhancedSequential, EnhancedSequential]:
  return (
    _build_vit(
      num_classes=num_classes,
      hidden_size=384,
      mlp_dim=1536,
      num_heads=6,
      patch_size=patch_size,
      num_layers=num_layers,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate,
      inference=False,
    ),
    _build_vit(
      num_classes=num_classes,
      hidden_size=384,
      mlp_dim=1536,
      num_heads=6,
      patch_size=patch_size,
      num_layers=num_layers,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate,
      inference=True,
    ),
  )
