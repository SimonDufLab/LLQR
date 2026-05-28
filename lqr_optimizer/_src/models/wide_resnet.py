import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

from lqr_optimizer._src.utils.utils import (
  EnhancedSequential,
  make_controlled_stage_descriptor,
  make_lqr_segment_descriptor,
  make_passive_stage_descriptor,
)


_WRN_DEPTH = 28
_WRN_WIDTH_FACTOR = 10
_WRN_DROPOUT_RATE = 0.0
_WRN_BN_EPSILON = 1e-5
_WRN_BN_MOMENTUM = 0.9
_WRN_FILTERS = (16, 16 * _WRN_WIDTH_FACTOR, 32 * _WRN_WIDTH_FACTOR, 64 * _WRN_WIDTH_FACTOR)
_WRN_BLOCK_DEPTH = (_WRN_DEPTH - 4) // 6


class ConvStage(nn.Module):
  features: int
  kernel_size: tuple = (3, 3)
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
      epsilon=_WRN_BN_EPSILON,
      momentum=_WRN_BN_MOMENTUM,
      name=self.bn_name,
    )(x)


class ReluStage(nn.Module):
  def __call__(self, x):
    return nn.relu(x)


class BasicUnitFirstBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, x):
    normalized = nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_WRN_BN_EPSILON,
      momentum=_WRN_BN_MOMENTUM,
      name=self.bn_name,
    )(x)
    return normalized, x


class DownsampleFirstBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, x):
    normalized = nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_WRN_BN_EPSILON,
      momentum=_WRN_BN_MOMENTUM,
      name=self.bn_name,
    )(x)
    return normalized, normalized


class TupleMainBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, inputs):
    x, carry = inputs
    return nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_WRN_BN_EPSILON,
      momentum=_WRN_BN_MOMENTUM,
      name=self.bn_name,
    )(x), carry


class TupleMainReluStage(nn.Module):
  def __call__(self, inputs):
    x, carry = inputs
    return nn.relu(x), carry


class TupleBothReluStage(nn.Module):
  def __call__(self, inputs):
    x, carry = inputs
    return nn.relu(x), nn.relu(carry)


class TupleMainConvStage(nn.Module):
  features: int
  kernel_size: tuple = (3, 3)
  strides: tuple = (1, 1)
  padding: str = "SAME"
  use_bias: bool = False
  conv_name: str = "Conv_0"

  @nn.compact
  def __call__(self, inputs):
    x, carry = inputs
    return nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(x), carry


class TupleDropoutStage(nn.Module):
  deterministic: bool
  rate: float = _WRN_DROPOUT_RATE

  @nn.compact
  def __call__(self, inputs):
    x, carry = inputs
    return nn.Dropout(rate=self.rate)(x, deterministic=self.deterministic), carry


class TupleSkipConvStage(nn.Module):
  features: int
  kernel_size: tuple = (1, 1)
  strides: tuple = (1, 1)
  padding: str = "VALID"
  use_bias: bool = False
  conv_name: str = "Conv_1"

  @nn.compact
  def __call__(self, inputs):
    x, carry = inputs
    return x, nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(carry)


class TupleAddStage(nn.Module):
  def __call__(self, inputs):
    x, carry = inputs
    return x + carry


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


def _append_basic_unit(layers, stage_descriptors, lqr_segment_descriptors, *, prefix, channels, inference):
  stage_names = (
    f"{prefix}_bn1",
    f"{prefix}_relu1",
    f"{prefix}_conv1",
    f"{prefix}_bn2",
    f"{prefix}_relu2",
    f"{prefix}_dropout",
    f"{prefix}_conv2",
    f"{prefix}_add",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_bn1",
    BasicUnitFirstBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  _append_passive(
    layers,
    stage_descriptors,
    f"{prefix}_relu1",
    TupleMainReluStage(),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv1",
    TupleMainConvStage(features=channels, strides=(1, 1), conv_name="Conv_0"),
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
  _append_passive(layers, stage_descriptors, f"{prefix}_dropout", TupleDropoutStage(deterministic=inference))
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv2",
    TupleMainConvStage(features=channels, strides=(1, 1), conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  _append_passive(layers, stage_descriptors, f"{prefix}_add", TupleAddStage(), passive_state_hessian="zero")
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      prefix,
      stage_names,
      sample_separable_second_order=bool(inference),
    )
  )


def _append_downsample_unit(layers, stage_descriptors, lqr_segment_descriptors, *, prefix, in_channels,
                            out_channels, stride, inference):
  stage_names = (
    f"{prefix}_bn1",
    f"{prefix}_relu1",
    f"{prefix}_conv1",
    f"{prefix}_bn2",
    f"{prefix}_relu2",
    f"{prefix}_dropout",
    f"{prefix}_conv2",
    f"{prefix}_skip_proj",
    f"{prefix}_add",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_bn1",
    DownsampleFirstBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
    fast_path_kind="linear_controlled" if inference else None,
  )
  _append_passive(
    layers,
    stage_descriptors,
    f"{prefix}_relu1",
    TupleBothReluStage(),
    fast_path_kind="piecewise_linear_passive",
    passive_state_hessian="zero",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv1",
    TupleMainConvStage(features=out_channels, strides=(stride, stride), conv_name="Conv_0"),
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
  _append_passive(layers, stage_descriptors, f"{prefix}_dropout", TupleDropoutStage(deterministic=inference))
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_conv2",
    TupleMainConvStage(features=out_channels, strides=(1, 1), conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  _append_controlled(
    layers,
    stage_descriptors,
    f"{prefix}_skip_proj",
    TupleSkipConvStage(features=out_channels, strides=(stride, stride), conv_name="Conv_1"),
    fast_path_kind="linear_controlled",
  )
  _append_passive(layers, stage_descriptors, f"{prefix}_add", TupleAddStage(), passive_state_hessian="zero")
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      prefix,
      stage_names,
      sample_separable_second_order=bool(inference),
    )
  )


def _build_wide_resnet28x10(inference: bool, num_classes: int) -> EnhancedSequential:
  layers = []
  stage_descriptors = []
  lqr_segment_descriptors = []

  _append_controlled(
    layers,
    stage_descriptors,
    "stem_conv",
    ConvStage(features=_WRN_FILTERS[0], strides=(1, 1), conv_name="Conv_0"),
    fast_path_kind="linear_controlled",
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor("stem", ("stem_conv",), sample_separable_second_order=bool(inference))
  )

  for block_index, (in_channels, out_channels, stride) in enumerate((
    (_WRN_FILTERS[0], _WRN_FILTERS[1], 1),
    (_WRN_FILTERS[1], _WRN_FILTERS[2], 2),
    (_WRN_FILTERS[2], _WRN_FILTERS[3], 2),
  )):
    _append_downsample_unit(
      layers,
      stage_descriptors,
      lqr_segment_descriptors,
      prefix=f"block_{block_index}_downsample",
      in_channels=in_channels,
      out_channels=out_channels,
      stride=stride,
      inference=inference,
    )
    for unit_index in range(1, _WRN_BLOCK_DEPTH + 1):
      _append_basic_unit(
        layers,
        stage_descriptors,
        lqr_segment_descriptors,
        prefix=f"block_{block_index}_unit_{unit_index}",
        channels=out_channels,
        inference=inference,
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


def create_wide_resnet28x10(num_classes: int) -> Tuple[EnhancedSequential, EnhancedSequential]:
  return _build_wide_resnet28x10(False, num_classes), _build_wide_resnet28x10(True, num_classes)
