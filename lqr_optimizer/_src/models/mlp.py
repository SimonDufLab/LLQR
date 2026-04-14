import flax.linen as nn
from typing import Tuple
from flax.core import freeze

from lqr_optimizer._src.utils.utils import (
  EnhancedSequential,
  make_controlled_stage_descriptor,
  make_passive_stage_descriptor,
)

class DenseRelu(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.channels)(x)
    return nn.relu(x)

class InitDenseRelu(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1)) if x.ndim == 4 else x.reshape((-1,))  # Flatten, assume BHWC images as input
    x = nn.Dense(self.channels)(x)
    return nn.relu(x)

class DenseLogSoftmax(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.channels)(x)
    x = nn.log_softmax(x)
    return x

class InitDenseStage(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1)) if x.ndim == 4 else x.reshape((-1,))
    return nn.Dense(self.channels)(x)


class DenseStage(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.channels)(x)


class ReluStage(nn.Module):
  def __call__(self, x):
    return nn.relu(x)


class LogSoftmaxStage(nn.Module):
  def __call__(self, x):
    return nn.log_softmax(x)


_LEGACY_MLP_PARAM_MAPPING = (
  ("layers_0", "layers_0"),
  ("layers_1", "layers_2"),
  ("layers_2", "layers_4"),
)


def _migrate_mlp_split_tree(loaded_tree, init_tree):
  if not init_tree and not loaded_tree:
    return loaded_tree
  if tuple(loaded_tree.keys()) == tuple(init_tree.keys()):
    return loaded_tree

  loaded_keys = tuple(loaded_tree.keys())
  expected_loaded_keys = tuple(old_key for old_key, _ in _LEGACY_MLP_PARAM_MAPPING)
  if loaded_keys != expected_loaded_keys:
    raise ValueError("Legacy MLP checkpoint layer count does not match the expected coarse-stage mapping.")

  migrated = {}
  for old_key, new_key in _LEGACY_MLP_PARAM_MAPPING:
    migrated[new_key] = loaded_tree[old_key]
  ordered = {key: migrated.get(key, init_tree[key]) for key in init_tree.keys()}
  return freeze(ordered)


def create_mlp_legacy(num_classes: int) -> Tuple[nn.Module, None]:
  layers = [InitDenseRelu(100), DenseRelu(300), DenseLogSoftmax(num_classes)]
  return EnhancedSequential(layers), None


def create_mlp(num_classes: int) -> Tuple[nn.Module, None]:
  layers = [
    InitDenseStage(100),
    ReluStage(),
    DenseStage(300),
    ReluStage(),
    DenseStage(num_classes),
    LogSoftmaxStage(),
  ]
  stage_descriptors = (
    make_controlled_stage_descriptor("dense0", "layers_0", fast_path_kind="linear_controlled"),
    make_passive_stage_descriptor("relu0", fast_path_kind="piecewise_linear_passive",
                                  passive_state_hessian="zero"),
    make_controlled_stage_descriptor("dense1", "layers_2", fast_path_kind="linear_controlled"),
    make_passive_stage_descriptor("relu1", fast_path_kind="piecewise_linear_passive",
                                  passive_state_hessian="zero"),
    make_controlled_stage_descriptor("logits", "layers_4", fast_path_kind="linear_controlled"),
    make_passive_stage_descriptor("log_softmax", passive_state_hessian="generic"),
  )

  def migrate_legacy_checkpoint(loaded_params, loaded_batch_stats, init_params, init_batch_stats):
    return (
      _migrate_mlp_split_tree(loaded_params, init_params),
      _migrate_mlp_split_tree(loaded_batch_stats, init_batch_stats),
    )

  model = EnhancedSequential(
    layers,
    stage_descriptors=stage_descriptors,
    legacy_checkpoint_migrator=migrate_legacy_checkpoint,
  )
  model.validate_stage_descriptors()
  return model, None

# class MLP(nn.Module):
#   num_classes: int
#
#   @nn.compact
#   def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#     # Create blocks for EnhancedSequential
#     blocks = [
#       nn.Sequential([nn.Dense(10), nn.relu]),
#       nn.Sequential([nn.Dense(128), nn.relu]),
#       nn.Sequential([nn.Dense(self.num_classes), nn.log_softmax]),  # Returns logits
#     ]
#
#     # Pass the blocks to EnhancedSequential
#     model = EnhancedSequential(blocks)
#     self._layers = model.layers
#
#     # Forward pass through EnhancedSequential
#     x = model(x)
#     return x
#
#   @property
#   def layer_names(self):
#     """Public getter for the stored layer names."""
#     return self._layer_names
#
#   @property
#   def layers(self):
#     """Public getter for the stored layer names."""
#     return self._layers
#
#   def init(self, rng: jax.random.PRNGKey, *args, **kwargs):
#     """
#     Overrides the init method to return the parameter dictionary
#     along with an ordered list of layer names based on the dictionary keys.
#
#     Args:
#         rng: A JAX random key for parameter initialization.
#         *args: Arguments to pass to the forward function.
#         **kwargs: Keyword arguments to pass to the forward function.
#
#     Returns:
#         - params: A FrozenDict containing the initialized parameters.
#         - layer_names: An ordered list of layer names from the params dictionary.
#     """
#     # Call the original init method to get parameters
#     params = super().init(rng, *args, **kwargs)
#
#     # Retrieve layer names directly from the parameter dictionary keys
#     self._layer_names = list(params["params"].keys())
#
#     return params
