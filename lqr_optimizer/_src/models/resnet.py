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

class StemBlock(nn.Module):
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

STARTING_FEATURES = 1 # Lowering down when debugging #TODO make configurable with hydra

def create_resnet18(num_classes: int) -> Tuple[EnhancedSequential, nn.Module]:
  def inference_mode(inference: bool):
    layers = []
    # Stem and max-pooling
    layers.append(StemBlock(inference=inference))
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