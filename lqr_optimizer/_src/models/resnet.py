import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple

from lqr_optimizer._src.utils.utils import EnhancedSequential


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
    # Stem and max-pooling
    layers.append(SmallStemBlock(inference=inference))
    # layers.append(MaxPool())

    # Group 1: two basic blocks with 64 filters (no downsampling)
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES, stride=1, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES, stride=1, inference=inference))
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES, stride=1, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES, stride=1, inference=inference))

    # Group 2: two basic blocks with 128 filters (first block downsamples)
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES*2, stride=2, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES*2, stride=2, inference=inference))
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES*2, stride=1, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES*2, stride=1, inference=inference))

    # Group 3: two basic blocks with 256 filters (first block downsamples)
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES*4, stride=2, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES*4, stride=2, inference=inference))
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES*4, stride=1, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES*4, stride=1, inference=inference))

    # Group 4: two basic blocks with 512 filters (first block downsamples)
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES*8, stride=2, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES*8, stride=2, inference=inference))
    layers.append(ResidualBlockPart1(features=STARTING_FEATURES*8, stride=1, inference=inference))
    layers.append(ResidualBlockPart2(features=STARTING_FEATURES*8, stride=1, inference=inference))

    # Global average pooling and final classification layer.
    # layers.append(GlobalAvgPool())
    layers.append(GPoolDenseLogSoftmax(num_classes))

    return EnhancedSequential(layers)

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

    # ImageNet stem (7×7/2 + BN + ReLU + 3×3 max-pool/2)
    layers.append(BigStemBlock(inference=inference))

    # Build stages conv2_x .. conv5_x
    for stage_idx, (planes, num_blocks) in enumerate(zip(STAGE_PLANES, STAGE_BLOCKS)):
      # Downsample on the first block of stages 2/3/4 (i.e., not on the very first stage)
      # Using v1.5: put stride on the 3×3 (BottleneckPart2).
      stride = 1 if stage_idx == 0 else 2

      # First block in the stage (may downsample)
      layers.append(BottleneckPart1(planes=planes, stride=1, inference=inference, expansion=EXPANSION))
      layers.append(BottleneckPart2(planes=planes, stride=stride, inference=inference, expansion=EXPANSION))
      layers.append(BottleneckPart3(planes=planes, stride=stride, inference=inference, expansion=EXPANSION))

      # Remaining blocks in the stage (no downsampling)
      for _ in range(num_blocks - 1):
        layers.append(BottleneckPart1(planes=planes, stride=1, inference=inference, expansion=EXPANSION))
        layers.append(BottleneckPart2(planes=planes, stride=1, inference=inference, expansion=EXPANSION))
        layers.append(BottleneckPart3(planes=planes, stride=1, inference=inference, expansion=EXPANSION))

    # Head: Global Average Pool → Dense(num_classes) → log_softmax
    layers.append(GPoolDenseLogSoftmax(num_classes))

    return EnhancedSequential(layers)

  return inference_mode(False), inference_mode(True)
