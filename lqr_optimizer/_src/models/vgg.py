import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

from lqr_optimizer._src.utils.utils import (
  EnhancedSequential,
  make_controlled_stage_descriptor,
  make_lqr_segment_descriptor,
  make_passive_stage_descriptor,
)


_VGG16_FEATURE_BLOCKS = (
  (64, 64),
  (128, 128),
  (256, 256, 256),
  (512, 512, 512),
  (512, 512, 512),
)
_VGG_BN_EPSILON = 1e-5
_VGG_BN_MOMENTUM = 0.9
_VGG_DROPOUT_RATE = 0.5


class ConvStage(nn.Module):
  features: int
  kernel_size: tuple = (3, 3)
  strides: tuple = (1, 1)
  padding: str = "SAME"
  use_bias: bool = True
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


class BatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, x):
    return nn.BatchNorm(
      use_running_average=self.inference,
      epsilon=_VGG_BN_EPSILON,
      momentum=_VGG_BN_MOMENTUM,
      name=self.bn_name,
    )(x)


class ReluStage(nn.Module):
  def __call__(self, x):
    return nn.relu(x)


class MaxPoolStage(nn.Module):
  def __call__(self, x):
    return nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")


class FlattenStage(nn.Module):
  def __call__(self, x):
    return jnp.reshape(x, (x.shape[0], -1))


class DropoutStage(nn.Module):
  deterministic: bool
  rate: float = _VGG_DROPOUT_RATE

  @nn.compact
  def __call__(self, x):
    return nn.Dropout(rate=self.rate)(x, deterministic=self.deterministic)


class DenseStage(nn.Module):
  features: int
  dense_name: str = "Dense_0"

  @nn.compact
  def __call__(self, x):
    return nn.Dense(features=self.features, name=self.dense_name)(x)


class LogSoftmaxStage(nn.Module):
  def __call__(self, x):
    return nn.log_softmax(x)


def _append_feature_block(layers, stage_descriptors, lqr_segment_descriptors, *, block_index, channels, inference):
  stage_names = []
  conv_index = 0
  for out_channels in channels:
    conv_name = f"block_{block_index}_conv_{conv_index}"
    bn_name = f"block_{block_index}_bn_{conv_index}"
    relu_name = f"block_{block_index}_relu_{conv_index}"
    layer_index = len(layers)
    layers.append(ConvStage(features=out_channels, conv_name="Conv_0"))
    stage_descriptors.append(
      make_controlled_stage_descriptor(conv_name, f"layers_{layer_index}", fast_path_kind="linear_controlled")
    )
    stage_names.append(conv_name)

    layer_index = len(layers)
    layers.append(BatchNormStage(inference=inference, bn_name="BatchNorm_0"))
    stage_descriptors.append(
      make_controlled_stage_descriptor(
        bn_name,
        f"layers_{layer_index}",
        fast_path_kind="linear_controlled" if inference else None,
      )
    )
    stage_names.append(bn_name)

    layers.append(ReluStage())
    stage_descriptors.append(
      make_passive_stage_descriptor(
        relu_name,
        fast_path_kind="piecewise_linear_passive",
        passive_state_hessian="zero",
      )
    )
    stage_names.append(relu_name)
    conv_index += 1

  pool_name = f"block_{block_index}_pool"
  layers.append(MaxPoolStage())
  stage_descriptors.append(make_passive_stage_descriptor(pool_name, passive_state_hessian="zero"))
  stage_names.append(pool_name)

  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      f"feature_block_{block_index}",
      tuple(stage_names),
      sample_separable_second_order=bool(inference),
    )
  )


def _build_vgg16bn(inference: bool, num_classes: int) -> EnhancedSequential:
  layers = []
  stage_descriptors = []
  lqr_segment_descriptors = []

  for block_index, channels in enumerate(_VGG16_FEATURE_BLOCKS):
    _append_feature_block(
      layers,
      stage_descriptors,
      lqr_segment_descriptors,
      block_index=block_index,
      channels=channels,
      inference=inference,
    )

  layers.append(FlattenStage())
  stage_descriptors.append(make_passive_stage_descriptor("flatten", passive_state_hessian="zero"))
  layers.append(DropoutStage(deterministic=inference))
  stage_descriptors.append(make_passive_stage_descriptor("dropout0"))
  layer_index = len(layers)
  layers.append(DenseStage(features=512, dense_name="Dense_0"))
  stage_descriptors.append(
    make_controlled_stage_descriptor("fc0", f"layers_{layer_index}", fast_path_kind="linear_controlled")
  )
  layers.append(ReluStage())
  stage_descriptors.append(
    make_passive_stage_descriptor("relu0", fast_path_kind="piecewise_linear_passive", passive_state_hessian="zero")
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      "classifier_block_0",
      ("flatten", "dropout0", "fc0", "relu0"),
      sample_separable_second_order=bool(inference),
    )
  )

  layers.append(DropoutStage(deterministic=inference))
  stage_descriptors.append(make_passive_stage_descriptor("dropout1"))
  layer_index = len(layers)
  layers.append(DenseStage(features=512, dense_name="Dense_0"))
  stage_descriptors.append(
    make_controlled_stage_descriptor("fc1", f"layers_{layer_index}", fast_path_kind="linear_controlled")
  )
  layers.append(ReluStage())
  stage_descriptors.append(
    make_passive_stage_descriptor("relu1", fast_path_kind="piecewise_linear_passive", passive_state_hessian="zero")
  )
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      "classifier_block_1",
      ("dropout1", "fc1", "relu1"),
      sample_separable_second_order=bool(inference),
    )
  )

  layer_index = len(layers)
  layers.append(DenseStage(features=num_classes, dense_name="Dense_0"))
  stage_descriptors.append(
    make_controlled_stage_descriptor("logits", f"layers_{layer_index}", fast_path_kind="linear_controlled")
  )
  layers.append(LogSoftmaxStage())
  stage_descriptors.append(make_passive_stage_descriptor("log_softmax", passive_state_hessian="generic"))
  lqr_segment_descriptors.append(
    make_lqr_segment_descriptor(
      "head",
      ("logits", "log_softmax"),
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


def create_vgg16bn(num_classes: int) -> Tuple[EnhancedSequential, EnhancedSequential]:
  return _build_vgg16bn(False, num_classes), _build_vgg16bn(True, num_classes)
