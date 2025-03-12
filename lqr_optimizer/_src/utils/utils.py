""" Various utilities functions for LQR optimization"""
import time
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import flax.linen as nn
from flax.linen import Sequential
from flax.core.frozen_dict import FrozenDict
from typing import List, Tuple, Any, Dict

def vjp_f(f, x):
  """ Return the vjp in a form that can be applied directly over a vector
  """
  _, f = jax.vjp(f, x)
  return lambda v: f(v)[0]


def add_f(f, g):
  """ Return the function composition of f added to g
  """
  return lambda x: f(x) + g(x)

def normalize_gradient(gradient):
  # Compute the total L2 norm of the gradient using jnp.linalg.norm
  total_norm = jnp.linalg.norm(ravel_pytree(gradient)[0], ord=2)

  # Avoid division by zero
  total_norm = jnp.maximum(total_norm, 1e-9)

  # Normalize each gradient component
  normalized_gradient = jax.tree_util.tree_map(lambda g: g / total_norm, gradient)

  return normalized_gradient

def clip_norm_single_example(_grad, clip_norm):
  """Apply clipping norm to a single example within a batch"""
  ravel_grad, unravel_fn = ravel_pytree(_grad)
  example_norm = jnp.linalg.norm(ravel_grad, ord=2)
  clipped_grad = ravel_grad * (clip_norm / jnp.maximum(example_norm, clip_norm))
  return unravel_fn(clipped_grad)

vmapped_clip_norm = jax.vmap(clip_norm_single_example, in_axes=(0, None))


##################################
# Flax utils
##################################
# class EnhancedSequential(Sequential):
#   def __init__(self, layers):
#     # Ensure proper initialization of layers
#     self._layer_names = None
#     super().__init__(layers)
#
#   @property
#   def layer_names(self):
#     return self._layer_names
#
#   def init(self, rng: jax.random.PRNGKey, *args, **kwargs) -> FrozenDict:
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

class EnhancedSequential(nn.Module):
  layers: List[nn.Module]

  def __call__(self, x: Any) -> Any:
    """Applies the blocks sequentially to the input."""
    for block in self.layers:
      x = block(x)
    return x

  def init(self, rng: jax.random.PRNGKey, *args, **kwargs) -> FrozenDict:
    """
    Overrides the init method to return the parameter dictionary
    along with an ordered list of layer names based on the dictionary keys.

    Args:
        rng: A JAX random key for parameter initialization.
        *args: Arguments to pass to the forward function.
        **kwargs: Keyword arguments to pass to the forward function.

    Returns:
        - params: A FrozenDict containing the initialized parameters.
        - layer_names: An ordered list of layer names from the params dictionary.
    """
    # Call the original init method to get parameters
    variables = super().init(rng, *args, **kwargs)

    return variables

  # def apply_block(self, block_name: str, x: Any, params: FrozenDict) -> Any:
  #   """Applies a specific block using its parameters."""
  #   block_params = params.get(block_name, {})
  #   for name, block in zip(list(params.keys()), self.layers):
  #     if name == block_name:
  #       return block.apply({"params": block_params}, x)
  #   raise ValueError(f"Block name '{block_name}' not found.")

  def apply_block_from_name(self, block_name: str, x: Any, params: FrozenDict) -> Any:
    """Applies a specific block using its parameters."""
    # Get the list of parameter names
    layer_names = list(params.keys())

    # Find the index of the block_name
    try:
      index = layer_names.index(block_name)
    except ValueError:
      raise ValueError(f"Block name '{block_name}' not found.")

    # Retrieve the corresponding block and its parameters
    block = self.layers[index]
    block_params = params.get(block_name, {})

    # Apply the block using the provided parameters
    return block.apply({"params": block_params}, x)

  def apply_block_from_params(self, block_params: FrozenDict, x: Any, index) -> Any:
    block = self.layers[index]
    return block.apply(block_params, x)

  ##################################
  # XLA debugging util
  ##################################
def timed_jit(f):
  """Wraps a function with JIT and times recompilation."""
  cache = {}

  def wrapped(*args):
    key = tuple(type(arg) for arg in args)  # Simple caching key based on input types
    if key not in cache:
      start_time = time.time()
      compiled_f = jax.jit(f)  # JIT compilation
      end_time = time.time()
      cache[key] = compiled_f
      print(f"Recompilation took {end_time - start_time:.6f} seconds")
    return cache[key](*args)

  return wrapped