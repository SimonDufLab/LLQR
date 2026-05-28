import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

from lqr_optimizer._src.utils.utils import (
  EnhancedSequential,
  make_controlled_stage_descriptor,
  make_lqr_segment_descriptor,
  make_passive_stage_descriptor,
)


_PYRAMIDNET_DEPTH = 110
_PYRAMIDNET_ALPHA = 270
_PYRAMIDNET_BLOCKS_PER_STAGE = (_PYRAMIDNET_DEPTH - 2) // 9
_PYRAMIDNET_BOTTLENECK_RATIO = 4
_PYRAMIDNET_STEM_CHANNELS = 16
_PYRAMIDNET_BN_EPSILON = 1e-5
_PYRAMIDNET_BN_MOMENTUM = 0.9
_PYRAMIDNET_ADDRATE = _PYRAMIDNET_ALPHA / (3 * _PYRAMIDNET_BLOCKS_PER_STAGE)


class ConvStage(nn.Module):
  features: int
  kernel_size: tuple
  strides: tuple = (1, 1)
  padding: str = "SAME"
  use_bias: bool = False
  conv_name: str = "Conv_0"

  @nn.compact
  def __call__(self, x):
    return nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(x)


class DenseStage(nn.Module):
  features: int
  use_bias: bool = True
  dense_name: str = "Dense_0"

  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=self.features, use_bias=self.use_bias, name=self.dense_name)(x)


class BatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, x):
    return nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_PYRAMIDNET_BN_EPSILON,
      momentum=_PYRAMIDNET_BN_MOMENTUM,
      name=self.bn_name,
    )(x)


class ReluStage(nn.Module):
  def __call__(self, x):
    return nn.relu(x)


class TupleCarryBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, x):
    normalized = nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_PYRAMIDNET_BN_EPSILON,
      momentum=_PYRAMIDNET_BN_MOMENTUM,
      name=self.bn_name,
    )(x)
    return normalized, x


class TupleMainBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, inputs):
    x, shortcut = inputs
    return nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_PYRAMIDNET_BN_EPSILON,
      momentum=_PYRAMIDNET_BN_MOMENTUM,
      name=self.bn_name,
    )(x), shortcut


class TupleMainReluStage(nn.Module):
  def __call__(self, inputs):
    x, shortcut = inputs
    return nn.relu(x), shortcut


class TupleMainConvStage(nn.Module):
  features: int
  kernel_size: tuple
  strides: tuple = (1, 1)
  padding: str = "SAME"
  use_bias: bool = False
  conv_name: str = "Conv_0"

  @nn.compact
  def __call__(self, inputs):
    x, shortcut = inputs
    return nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(x), shortcut


class TupleShortcutAvgPoolStage(nn.Module):
  def __call__(self, inputs):
    x, shortcut = inputs
    return x, nn.avg_pool(shortcut, window_shape=(2, 2), strides=(2, 2), padding="VALID")


class TupleShortcutPadStage(nn.Module):
  channels_to_add: int

  def __call__(self, inputs):
    x, shortcut = inputs
    if self.channels_to_add == 0:
      return x, shortcut
    return x, jnp.pad(shortcut, ((0, 0), (0, 0), (0, 0), (0, self.channels_to_add)))


class TupleAddStage(nn.Module):
  def __call__(self, inputs):
    x, shortcut = inputs
    return x + shortcut


class AvgPoolStage(nn.Module):
  def __call__(self, x):
    return nn.avg_pool(x, window_shape=(8, 8), strides=(8, 8), padding="VALID")


class FlattenStage(nn.Module):
  def __call__(self, x):
    return jnp.reshape(x, (x.shape[0], -1))


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


def _iter_pyramidnet_units():
  featuremap_dim = float(_PYRAMIDNET_STEM_CHANNELS)
  input_featuremap_dim = _PYRAMIDNET_STEM_CHANNELS

  for block_index in range(3):
    for unit_index in range(_PYRAMIDNET_BLOCKS_PER_STAGE):
      stride = 2 if block_index > 0 and unit_index == 0 else 1
      featuremap_dim += _PYRAMIDNET_ADDRATE
      planes = int(round(featuremap_dim))
      out_channels = planes * _PYRAMIDNET_BOTTLENECK_RATIO
      yield {
        "prefix": f"block_{block_index}_unit_{unit_index}",
        "planes": planes,
        "stride": stride,
        "out_channels": out_channels,
        "shortcut_pool": stride != 1,
        "shortcut_pad": out_channels - input_featuremap_dim,
      }
      input_featuremap_dim = out_channels


def _append_bottleneck_unit(layers, stage_descriptors, lqr_segment_descriptors, *, prefix, planes, stride,
                            out_channels, shortcut_pool, shortcut_pad, inference):
  stage_names = [
    f"{prefix}_bn1",
    f"{prefix}_conv1",
    f"{prefix}_bn2",
    f"{prefix}_relu2",
    f"{prefix}_conv2",
    f"{prefix}_bn3",
    f"{prefix}_relu3",
    f"{prefix}_conv3",
    f"{prefix}_bn4",
  ]

  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_bn1",
    TupleCarryBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv1",
    TupleMainConvStage(features=planes, kernel_size=(1, 1), strides=(1, 1), padding="VALID", conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_bn2",
    TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  _append_passive(
    layers,
    stage_descriptors,
    f"{prefix}_relu2",
    TupleMainReluStage(),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv2",
    TupleMainConvStage(features=planes, kernel_size=(3, 3), strides=(stride, stride), padding="SAME",
                       conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_bn3",
    TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  _append_passive(
    layers,
    stage_descriptors,
    f"{prefix}_relu3",
    TupleMainReluStage(),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv3",
    TupleMainConvStage(features=out_channels, kernel_size=(1, 1), strides=(1, 1), padding="VALID",
                       conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_bn4",
    TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )

  if shortcut_pool:
    _append_passive(
      layers,
      stage_descriptors,
      f"{prefix}_shortcut_pool",
      TupleShortcutAvgPoolStage(),
      passive_state_hessian="zero",
    )
    stage_names.append(f"{prefix}_shortcut_pool")

  if shortcut_pad:
    _append_passive(
      layers,
      stage_descriptors,
      f"{prefix}_shortcut_pad",
      TupleShortcutPadStage(channels_to_add=shortcut_pad),
      passive_state_hessian="zero",
    )
    stage_names.append(f"{prefix}_shortcut_pad")

  _append_passive(layers, stage_descriptors, f"{prefix}_add", TupleAddStage(), passive_state_hessian="zero")
  stage_names.append(f"{prefix}_add")
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      prefix,
      tuple(stage_names),
      sample_separable_second_order=bool(inference),
    )
  )


def _build_pyramidnet110(inference: bool, num_classes: int) -> EnhancedSequential:
  layers = []
  stage_descriptors = []
  lqr_segment_descriptors = []

  _append_controlled(
    layers,
    stage_descriptors,
    "stem_conv",
    ConvStage(features=_PYRAMIDNET_STEM_CHANNELS, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
              conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    "stem_bn",
    BatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor("stem", ("stem_conv", "stem_bn"), sample_separable_second_order=bool(inference))
  )

  for unit_spec in _iter_pyramidnet_units():
    _append_bottleneck_unit(
      layers,
      stage_descriptors,
      lqr_segment_descriptors,
      inference=inference,
      **unit_spec,
    )

  _append_controlled(
    layers,
    stage_descriptors,
    "head_bn",
    BatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  _append_passive(
    layers,
    stage_descriptors,
    "head_relu",
    ReluStage(),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_passive(layers, stage_descriptors, "head_pool", AvgPoolStage(), passive_state_hessian="zero")
  _append_passive(layers, stage_descriptors, "flatten", FlattenStage(), passive_state_hessian="zero")
  _append_controlled(
    layers,
    stage_descriptors,
    "logits",
    DenseStage(features=num_classes, dense_name="Dense_0"),
    fast_path_kind="linear_controlled",
  )
  _append_passive(layers, stage_descriptors, "log_softmax", LogSoftmaxStage(), passive_state_hessian="generic")
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      "head",
      ("head_bn", "head_relu", "head_pool", "flatten", "logits", "log_softmax"),
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


def create_pyramidnet110(num_classes: int) -> Tuple[EnhancedSequential, EnhancedSequential]:
  return _build_pyramidnet110(False, num_classes), _build_pyramidnet110(True, num_classes)
