import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

from flax.core import freeze

from lqr_optimizer._src.utils.utils import EnhancedSequential, StageDescriptor


# ============================================================================
# Modules that each contain at most one convolution in their main branch.
# To support skip connections in a BasicBlock we “augment” the data:
#
#   • The first half-block takes a tensor x, applies a 3×3 conv–bn–relu, and
#     returns a tuple (out, identity) where identity is the original x.
#
#   • The second half-block accepts that tuple, applies a 3×3 conv–bn, and then
#     (if needed) projects identity before adding and applying relu.
# ============================================================================

class BigStemBlock(nn.Module): # Original version, better suited to ImageNet
  """Initial stem: 7×7 conv, batch norm, ReLU and MaxPool."""
  inference: bool = False

  @nn.compact
  def __call__(self, x):
    # One conv here (with bn and relu grouped) counts as one block.
    x = nn.Conv(features=64, kernel_size=(7, 7), strides=(2, 2),
                padding='SAME', use_bias=False)(x)
    x = nn.BatchNorm(use_running_average=self.inference)(x)
    x = nn.relu(x)
    x = nn.max_pool(x, window_shape=(3,3),
                       strides=(2,2), padding='SAME')
    return x


class SmallStemBlock(nn.Module):  # Version  for  Cifar-10/100, better suited to smaller img resolution
  inference: bool = False

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', use_bias=False)(x)
    x = nn.BatchNorm(use_running_average=self.inference)(x)
    return nn.relu(x)


# class MaxPool(nn.Module):
#   """A simple max-pooling block."""
#   window_shape: tuple = (3, 3)
#   strides: tuple = (2, 2)
#   padding: str = 'SAME'
#
#   def __call__(self, x):
#     return nn.max_pool(x, window_shape=self.window_shape,
#                        strides=self.strides, padding=self.padding)


class ResidualBlockPart1(nn.Module):
  """
  The first half of a basic block.

  This module applies a 3×3 conv–bn–relu (with the specified stride) to x and
  returns a tuple (out, identity) where identity is the input that will later be
  added in the second half-block.
  """
  features: int
  stride: int = 1
  inference: bool = False

  @nn.compact
  def __call__(self, x):
    identity = x
    out = nn.Conv(features=self.features, kernel_size=(3, 3),
                  strides=(self.stride, self.stride), padding='SAME',
                  use_bias=False)(x)
    out = nn.BatchNorm(use_running_average=self.inference)(out)
    out = nn.relu(out)
    # Return the convolution output and the “skip” (the unmodified input)
    return (out, identity)


class ResidualBlockPart2(nn.Module):
  """
  The second half of a basic block.

  This module expects as input a tuple (x, identity). It applies a 3×3 conv–bn
  (with stride 1) to x. If the shape of identity doesn’t match, a projection is
  applied to identity. Then the two are added and the result is activated.
  """
  features: int
  # We pass the same stride used in Part1 so that, in downsampling cases,
  # the projection on the identity uses that stride.
  stride: int = 1
  inference: bool = False

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    out = nn.Conv(features=self.features, kernel_size=(3, 3),
                  strides=(1, 1), padding='SAME', use_bias=False)(x)
    out = nn.BatchNorm(use_running_average=self.inference)(out)
    # If shapes differ (e.g. because of downsampling) project identity.
    if identity.shape != out.shape:
      identity = nn.Conv(features=self.features, kernel_size=(1, 1),
                         strides=(self.stride, self.stride),
                         padding='SAME', use_bias=False)(identity)
      identity = nn.BatchNorm(use_running_average=self.inference)(identity)
    out = out + identity
    out = nn.relu(out)
    return out


class GlobalAvgPool(nn.Module):
  """Global average pooling over spatial dimensions."""

  def __call__(self, x):
    return jnp.mean(x, axis=(-3, -2)) # Using negative index for compatibility in preconditioner vmap

class GPoolDenseLogSoftmax(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = GlobalAvgPool()(x)
    x = nn.Dense(features=self.channels)(x)
    x = nn.log_softmax(x)
    return x


class ConvStage(nn.Module):
  features: int
  kernel_size: tuple
  strides: tuple = (1, 1)
  padding: str = 'SAME'
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


class BatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, x):
    return nn.BatchNorm(use_running_average=self.inference, name=self.bn_name)(x)


class ReluStage(nn.Module):
  def __call__(self, x):
    return nn.relu(x)


class MaxPoolStage(nn.Module):
  window_shape: tuple = (3, 3)
  strides: tuple = (2, 2)
  padding: str = 'SAME'

  def __call__(self, x):
    return nn.max_pool(x, window_shape=self.window_shape, strides=self.strides, padding=self.padding)


class CarryIdentityConvStage(nn.Module):
  features: int
  kernel_size: tuple
  strides: tuple = (1, 1)
  padding: str = 'SAME'
  use_bias: bool = False
  conv_name: str = "Conv_0"

  @nn.compact
  def __call__(self, x):
    conv_out = nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(x)
    return conv_out, x


class TupleMainBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_0"

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    return nn.BatchNorm(use_running_average=self.inference, name=self.bn_name)(x), identity


class TupleMainConvStage(nn.Module):
  features: int
  kernel_size: tuple
  strides: tuple = (1, 1)
  padding: str = 'SAME'
  use_bias: bool = False
  conv_name: str = "Conv_0"

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    conv_out = nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(x)
    return conv_out, identity


class TupleSkipConvStage(nn.Module):
  features: int
  kernel_size: tuple = (1, 1)
  strides: tuple = (1, 1)
  padding: str = 'SAME'
  use_bias: bool = False
  conv_name: str = "Conv_1"

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    projected = nn.Conv(
      features=self.features,
      kernel_size=self.kernel_size,
      strides=self.strides,
      padding=self.padding,
      use_bias=self.use_bias,
      name=self.conv_name,
    )(identity)
    return x, projected


class TupleSkipBatchNormStage(nn.Module):
  inference: bool = False
  bn_name: str = "BatchNorm_1"

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    return x, nn.BatchNorm(use_running_average=self.inference, name=self.bn_name)(identity)


class TupleReluStage(nn.Module):
  def __call__(self, inputs):
    x, identity = inputs
    return nn.relu(x), identity


class TupleAddReluStage(nn.Module):
  def __call__(self, inputs):
    x, identity = inputs
    return nn.relu(x + identity)


def _extract_migrated_layer(source_layer, subkeys):
  if subkeys is None:
    return source_layer
  return {subkey: source_layer[subkey] for subkey in subkeys if subkey in source_layer}


def _migrate_split_stage_tree(loaded_tree, init_tree, legacy_mapping):
  if not init_tree and not loaded_tree:
    return loaded_tree
  if tuple(loaded_tree.keys()) == tuple(init_tree.keys()):
    return loaded_tree

  loaded_keys = list(loaded_tree.keys())
  if len(loaded_keys) != len(legacy_mapping):
    raise ValueError("Legacy checkpoint layer count does not match the expected coarse-stage mapping.")

  migrated = {}
  for old_key, split_targets in zip(loaded_keys, legacy_mapping):
    source_layer = loaded_tree[old_key]
    for new_key, subkeys in split_targets:
      migrated[new_key] = _extract_migrated_layer(source_layer, subkeys)

  ordered = {key: migrated.get(key, init_tree[key]) for key in init_tree.keys()}
  return freeze(ordered)

# ============================================================================
# Now we build ResNet-18 as an EnhancedSequential.
#
# The overall structure is:
#
#   StemBlock (with MaxPool) →
#
#   Group 1 (2 basic blocks, 64 filters, stride 1):
#       • ResidualBlockPart1(64, stride=1)
#       • ResidualBlockPart2(64, stride=1)
#       • ResidualBlockPart1(64, stride=1)
#       • ResidualBlockPart2(64, stride=1)
#
#   Group 2 (2 basic blocks, 128 filters, first block stride 2, second stride 1):
#       • ResidualBlockPart1(128, stride=2)
#       • ResidualBlockPart2(128, stride=2)
#       • ResidualBlockPart1(128, stride=1)
#       • ResidualBlockPart2(128, stride=1)
#
#   Group 3 (2 basic blocks, 256 filters, first block stride 2, second stride 1):
#       • ResidualBlockPart1(256, stride=2)
#       • ResidualBlockPart2(256, stride=2)
#       • ResidualBlockPart1(256, stride=1)
#       • ResidualBlockPart2(256, stride=1)
#
#   Group 4 (2 basic blocks, 512 filters, first block stride 2, second stride 1):
#       • ResidualBlockPart1(512, stride=2)
#       • ResidualBlockPart2(512, stride=2)
#       • ResidualBlockPart1(512, stride=1)
#       • ResidualBlockPart2(512, stride=1)
#
#   GlobalAvgPool → Dense(num_classes)
#
# Notice that every module in the sequential list has at most one conv in its
# “main branch”. The skip‐connection logic is implemented by having the first
# half-block return a tuple (output, identity) which is then “consumed” by the
# second half-block.
# ============================================================================

STARTING_FEATURES = 64 # Lowering down when debugging #TODO make configurable with hydra

def create_resnet18(num_classes: int) -> Tuple[EnhancedSequential, nn.Module]:
  def inference_mode(inference: bool):
    layers = []
    stage_descriptors = []
    legacy_mapping = []
    legacy_batch_stats_mapping = []

    def add_controlled(stage_name, module, fast_path_kind=None):
      stage_index = len(layers)
      layers.append(module)
      param_name = f"layers_{stage_index}"
      stage_descriptors.append(StageDescriptor(stage_name, "controlled", param_name, fast_path_kind))
      return param_name

    def add_passive(stage_name, module, fast_path_kind=None):
      layers.append(module)
      stage_descriptors.append(StageDescriptor(stage_name, "passive", None, fast_path_kind))

    def append_basic_block(prefix, features, stride, projection):
      part1_mapping = []
      part2_mapping = []
      part1_batch_stats_mapping = []
      part2_batch_stats_mapping = []
      part1_mapping.append((add_controlled(f"{prefix}_conv1",
                                           CarryIdentityConvStage(features=features, kernel_size=(3, 3),
                                                                  strides=(stride, stride), conv_name="Conv_0"),
                                           fast_path_kind="linear_controlled"),
                            ("Conv_0",)))
      bn1_key = add_controlled(f"{prefix}_bn1",
                               TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                               fast_path_kind="linear_controlled" if inference else None)
      part1_mapping.append((bn1_key, ("BatchNorm_0",)))
      part1_batch_stats_mapping.append((bn1_key, ("BatchNorm_0",)))
      add_passive(f"{prefix}_relu1", TupleReluStage(), fast_path_kind="piecewise_linear_passive")
      part2_mapping.append((add_controlled(f"{prefix}_conv2",
                                           TupleMainConvStage(features=features, kernel_size=(3, 3),
                                                              strides=(1, 1), conv_name="Conv_0"),
                                           fast_path_kind="linear_controlled"),
                            ("Conv_0",)))
      bn2_key = add_controlled(f"{prefix}_bn2",
                               TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                               fast_path_kind="linear_controlled" if inference else None)
      part2_mapping.append((bn2_key, ("BatchNorm_0",)))
      part2_batch_stats_mapping.append((bn2_key, ("BatchNorm_0",)))
      if projection:
        part2_mapping.append((add_controlled(f"{prefix}_skip_proj_conv",
                                             TupleSkipConvStage(features=features, strides=(stride, stride),
                                                                conv_name="Conv_1"),
                                             fast_path_kind="linear_controlled"),
                              ("Conv_1",)))
        skip_bn_key = add_controlled(f"{prefix}_skip_proj_bn",
                                     TupleSkipBatchNormStage(inference=inference, bn_name="BatchNorm_1"),
                                     fast_path_kind="linear_controlled" if inference else None)
        part2_mapping.append((skip_bn_key, ("BatchNorm_1",)))
        part2_batch_stats_mapping.append((skip_bn_key, ("BatchNorm_1",)))
      add_passive(f"{prefix}_add_relu", TupleAddReluStage(), fast_path_kind="piecewise_linear_passive")
      return (
        tuple(part1_mapping),
        tuple(part2_mapping),
        tuple(part1_batch_stats_mapping),
        tuple(part2_batch_stats_mapping),
      )

    stem_conv_key = add_controlled("stem_conv", ConvStage(features=64, kernel_size=(3, 3), strides=(1, 1), conv_name="Conv_0"),
                                   fast_path_kind="linear_controlled")
    stem_bn_key = add_controlled("stem_bn", BatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                                 fast_path_kind="linear_controlled" if inference else None)
    legacy_mapping.append(((stem_conv_key, ("Conv_0",)), (stem_bn_key, ("BatchNorm_0",))))
    legacy_batch_stats_mapping.append(((stem_bn_key, ("BatchNorm_0",)),))
    add_passive("stem_relu", ReluStage(), fast_path_kind="piecewise_linear_passive")

    block_specs = [
      (STARTING_FEATURES, 1, False),
      (STARTING_FEATURES, 1, False),
      (STARTING_FEATURES * 2, 2, True),
      (STARTING_FEATURES * 2, 1, False),
      (STARTING_FEATURES * 4, 2, True),
      (STARTING_FEATURES * 4, 1, False),
      (STARTING_FEATURES * 8, 2, True),
      (STARTING_FEATURES * 8, 1, False),
    ]

    for block_index, (features, stride, projection) in enumerate(block_specs):
      part1_mapping, part2_mapping, part1_batch_stats_mapping, part2_batch_stats_mapping = append_basic_block(
        f"block_{block_index}", features, stride, projection
      )
      legacy_mapping.append(part1_mapping)
      legacy_mapping.append(part2_mapping)
      legacy_batch_stats_mapping.append(part1_batch_stats_mapping)
      legacy_batch_stats_mapping.append(part2_batch_stats_mapping)

    legacy_mapping.append(((add_controlled("head", GPoolDenseLogSoftmax(num_classes)), None),))

    def migrate_legacy_checkpoint(loaded_params, loaded_batch_stats, init_params, init_batch_stats):
      return (
        _migrate_split_stage_tree(loaded_params, init_params, legacy_mapping),
        _migrate_split_stage_tree(loaded_batch_stats, init_batch_stats, legacy_batch_stats_mapping),
      )

    return EnhancedSequential(
      layers,
      stage_descriptors=tuple(stage_descriptors),
      legacy_checkpoint_migrator=migrate_legacy_checkpoint,
    )

  return inference_mode(False), inference_mode(True)

# ─────────────────────────────────────────────────────────────────────────────
# Bottleneck (ResNet-50/101) split into 3 modules, each with a single conv.
# Part1: 1×1 reduce + BN + ReLU → returns (out, identity)
# Part2: 3×3 (+stride if downsampling) + BN + ReLU → returns (out, identity)
# Part3: 1×1 expand + BN, optional proj on identity (1×1, stride), add + ReLU
# We implement ResNet-50 v1.5: the stride (2) is placed on the 3×3 conv.
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckPart1(nn.Module):
  """1×1 reduce, keep identity aside."""
  planes: int            # bottleneck width before expansion
  stride: int = 1        # kept for API symmetry; not used here
  inference: bool = False
  expansion: int = 4

  @nn.compact
  def __call__(self, x):
    identity = x
    out = nn.Conv(features=self.planes, kernel_size=(1, 1), strides=(1, 1),
                  padding='SAME', use_bias=False)(x)
    out = nn.BatchNorm(use_running_average=self.inference)(out)
    out = nn.relu(out)
    return (out, identity)


class BottleneckPart2(nn.Module):
  """3×3 conv (with stride for downsampling in v1.5), keep identity aside."""
  planes: int
  stride: int = 1
  inference: bool = False
  expansion: int = 4

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    out = nn.Conv(features=self.planes, kernel_size=(3, 3),
                  strides=(self.stride, self.stride),
                  padding='SAME', use_bias=False)(x)
    out = nn.BatchNorm(use_running_average=self.inference)(out)
    out = nn.relu(out)
    return (out, identity)


class BottleneckPart3(nn.Module):
  """1×1 expand, project identity if needed, add + ReLU."""
  planes: int
  stride: int = 1
  inference: bool = False
  expansion: int = 4

  @nn.compact
  def __call__(self, inputs):
    x, identity = inputs
    out_channels = self.planes * self.expansion

    out = nn.Conv(features=out_channels, kernel_size=(1, 1),
                  strides=(1, 1), padding='SAME', use_bias=False)(x)
    out = nn.BatchNorm(use_running_average=self.inference)(out)

    # Project identity if spatial dims or channel dims differ.
    if identity.shape != out.shape:
      identity = nn.Conv(features=out_channels, kernel_size=(1, 1),
                         strides=(self.stride, self.stride),
                         padding='SAME', use_bias=False)(identity)
      identity = nn.BatchNorm(use_running_average=self.inference)(identity)

    out = out + identity
    out = nn.relu(out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-50 creation (ImageNet): stem → [3,4,6,3] bottleneck blocks →
# GlobalAvgPool → Dense(num_classes) → log_softmax
# We keep your GPoolDenseLogSoftmax for symmetry with CIFAR code.
# If you prefer raw logits for loss, you can swap GPoolDenseLogSoftmax
# for GlobalAvgPool() + nn.Dense(num_classes) in your pipeline.
# ─────────────────────────────────────────────────────────────────────────────

def create_resnet50(num_classes: int) -> Tuple[EnhancedSequential, nn.Module]:
  """Build ResNet-50 (v1.5) using the tuple-carry split bottleneck logic."""
  EXPANSION = 4
  STAGE_PLANES = [64, 128, 256, 512]   # bottleneck widths before expansion
  STAGE_BLOCKS = [3, 4, 6, 3]          # number of bottleneck blocks per stage

  def inference_mode(inference: bool):
    layers = []
    stage_descriptors = []
    legacy_mapping = []
    legacy_batch_stats_mapping = []

    def add_controlled(stage_name, module, fast_path_kind=None):
      stage_index = len(layers)
      layers.append(module)
      param_name = f"layers_{stage_index}"
      stage_descriptors.append(StageDescriptor(stage_name, "controlled", param_name, fast_path_kind))
      return param_name

    def add_passive(stage_name, module, fast_path_kind=None):
      layers.append(module)
      stage_descriptors.append(StageDescriptor(stage_name, "passive", None, fast_path_kind))

    def append_bottleneck(prefix, planes, stride, projection):
      part1_mapping = []
      part2_mapping = []
      part3_mapping = []
      part1_batch_stats_mapping = []
      part2_batch_stats_mapping = []
      part3_batch_stats_mapping = []
      out_channels = planes * EXPANSION
      part1_mapping.append((add_controlled(f"{prefix}_conv1",
                                           CarryIdentityConvStage(features=planes, kernel_size=(1, 1),
                                                                  strides=(1, 1), conv_name="Conv_0"),
                                           fast_path_kind="linear_controlled"),
                            ("Conv_0",)))
      bn1_key = add_controlled(f"{prefix}_bn1",
                               TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                               fast_path_kind="linear_controlled" if inference else None)
      part1_mapping.append((bn1_key, ("BatchNorm_0",)))
      part1_batch_stats_mapping.append((bn1_key, ("BatchNorm_0",)))
      add_passive(f"{prefix}_relu1", TupleReluStage(), fast_path_kind="piecewise_linear_passive")
      part2_mapping.append((add_controlled(f"{prefix}_conv2",
                                           TupleMainConvStage(features=planes, kernel_size=(3, 3),
                                                              strides=(stride, stride), conv_name="Conv_0"),
                                           fast_path_kind="linear_controlled"),
                            ("Conv_0",)))
      bn2_key = add_controlled(f"{prefix}_bn2",
                               TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                               fast_path_kind="linear_controlled" if inference else None)
      part2_mapping.append((bn2_key, ("BatchNorm_0",)))
      part2_batch_stats_mapping.append((bn2_key, ("BatchNorm_0",)))
      add_passive(f"{prefix}_relu2", TupleReluStage(), fast_path_kind="piecewise_linear_passive")
      part3_mapping.append((add_controlled(f"{prefix}_conv3",
                                           TupleMainConvStage(features=out_channels, kernel_size=(1, 1),
                                                              strides=(1, 1), conv_name="Conv_0"),
                                           fast_path_kind="linear_controlled"),
                            ("Conv_0",)))
      bn3_key = add_controlled(f"{prefix}_bn3",
                               TupleMainBatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                               fast_path_kind="linear_controlled" if inference else None)
      part3_mapping.append((bn3_key, ("BatchNorm_0",)))
      part3_batch_stats_mapping.append((bn3_key, ("BatchNorm_0",)))
      if projection:
        part3_mapping.append((add_controlled(f"{prefix}_skip_proj_conv",
                                             TupleSkipConvStage(features=out_channels, strides=(stride, stride),
                                                                conv_name="Conv_1"),
                                             fast_path_kind="linear_controlled"),
                              ("Conv_1",)))
        skip_bn_key = add_controlled(f"{prefix}_skip_proj_bn",
                                     TupleSkipBatchNormStage(inference=inference, bn_name="BatchNorm_1"),
                                     fast_path_kind="linear_controlled" if inference else None)
        part3_mapping.append((skip_bn_key, ("BatchNorm_1",)))
        part3_batch_stats_mapping.append((skip_bn_key, ("BatchNorm_1",)))
      add_passive(f"{prefix}_add_relu", TupleAddReluStage(), fast_path_kind="piecewise_linear_passive")
      return (
        tuple(part1_mapping),
        tuple(part2_mapping),
        tuple(part3_mapping),
        tuple(part1_batch_stats_mapping),
        tuple(part2_batch_stats_mapping),
        tuple(part3_batch_stats_mapping),
      )

    stem_conv_key = add_controlled("stem_conv", ConvStage(features=64, kernel_size=(7, 7), strides=(2, 2), conv_name="Conv_0"),
                                   fast_path_kind="linear_controlled")
    stem_bn_key = add_controlled("stem_bn", BatchNormStage(inference=inference, bn_name="BatchNorm_0"),
                                 fast_path_kind="linear_controlled" if inference else None)
    legacy_mapping.append(((stem_conv_key, ("Conv_0",)), (stem_bn_key, ("BatchNorm_0",))))
    legacy_batch_stats_mapping.append(((stem_bn_key, ("BatchNorm_0",)),))
    add_passive("stem_relu", ReluStage(), fast_path_kind="piecewise_linear_passive")
    add_passive("stem_pool", MaxPoolStage())

    block_id = 0
    for stage_idx, (planes, num_blocks) in enumerate(zip(STAGE_PLANES, STAGE_BLOCKS)):
      stride = 1 if stage_idx == 0 else 2
      bottleneck_mappings = append_bottleneck(f"block_{block_id}", planes, stride, True)
      legacy_mapping.extend(bottleneck_mappings[:3])
      legacy_batch_stats_mapping.extend(bottleneck_mappings[3:])
      block_id += 1
      for _ in range(num_blocks - 1):
        bottleneck_mappings = append_bottleneck(f"block_{block_id}", planes, 1, False)
        legacy_mapping.extend(bottleneck_mappings[:3])
        legacy_batch_stats_mapping.extend(bottleneck_mappings[3:])
        block_id += 1

    legacy_mapping.append(((add_controlled("head", GPoolDenseLogSoftmax(num_classes)), None),))

    def migrate_legacy_checkpoint(loaded_params, loaded_batch_stats, init_params, init_batch_stats):
      return (
        _migrate_split_stage_tree(loaded_params, init_params, legacy_mapping),
        _migrate_split_stage_tree(loaded_batch_stats, init_batch_stats, legacy_batch_stats_mapping),
      )

    return EnhancedSequential(
      layers,
      stage_descriptors=tuple(stage_descriptors),
      legacy_checkpoint_migrator=migrate_legacy_checkpoint,
    )

  return inference_mode(False), inference_mode(True)
