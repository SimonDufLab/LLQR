"""A flax implementation of single-layer functions Rosenbrock, Ackley and Goldstein-Price
"""
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from flax.linen.initializers import lecun_uniform
from flax.linen.module import Module, compact

from lqr_optimizer._src.utils.utils import EnhancedSequential

PRNGKey = Any
Shape = Tuple[int]
Dtype = Any
Array = Any

class Rosenbrock2DBlock(Module):
  """Consider parameter (a,b) as input an apply Rosenbrock function"""
  dtype: Any = jnp.float32
  precision: Any = None

  param_init: Callable[[PRNGKey, Shape, Dtype], Array] = lecun_uniform

  @compact
  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    params = self.param("kernel", self.param_init(), (1, 2))
    params = jnp.asarray(params, self.dtype)
    inputs = inputs.flatten()
    params = params.flatten()
    y = inputs[1] * (params[1] - params[0] ** 2) ** 2 + (inputs[0] - params[0]) ** 2
    return y

class RosenbrockIterator:
  def __init__(self, a=1.0, b=100.0, batch_size=1):
    """
    Initialize the Rosenbrock iterator.

    Args:
        a (float): Parameter 'a' of the Rosenbrock function.
        b (float): Parameter 'b' of the Rosenbrock function.
        batch_size (int): Number of copies of [a, b] per batch.
    """
    self.a = a
    self.b = b
    self.batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self):
    """
    Generate the next batch of parameters for the Rosenbrock function.

    Returns:
        tuple: A tuple containing a numpy array of shape (batch_size, 2)
               where each row is [a, b], and None.
    """
    batch = jnp.array([[self.a, self.b]] * self.batch_size)
    return batch, None

def get_rosenbrock_model_and_datagen(a=1.0, b=100.0):
  return EnhancedSequential([Rosenbrock2DBlock(),]), RosenbrockIterator(a, b)

class AckleyBlock(Module):
  """Consider parameter (a,b,c) as input to apply Ackley function"""
  dimension: int = 3
  dtype: Any = jnp.float32
  precision: Any = None

  param_init: Callable[[PRNGKey, Shape, Dtype], Array] = lecun_uniform

  @compact
  def __call__(self, inputs: Array) -> Array:
    inputs = jnp.asarray(inputs, self.dtype)
    params = self.param("kernel", self.param_init(), (1, self.dimension))
    params = jnp.asarray(params, self.dtype)
    inputs = inputs.flatten()
    params = params.flatten()
    y = -inputs[0] * jnp.exp(-inputs[1] * jnp.sqrt(jnp.sum(params ** 2) / self.dimension)) - jnp.exp(
      jnp.sum(jnp.cos(inputs[2] * params)) / self.dimension) + inputs[0] + jnp.exp(1.0)
    return y

class AckleyIterator:
  def __init__(self, a=20.0, b=0.2, c=2*jnp.pi, batch_size=1):
    """
    Initialize the Ackley iterator.

    Args:
        a (float): Parameter 'a' of the Ackley function.
        b (float): Parameter 'b' of the Ackley function.
        c (float): Parameter 'c' of the Ackley function.
        batch_size (int): Number of copies of [a, b, c] per batch.
    """
    self.a = a
    self.b = b
    self.c = c
    self.batch_size = batch_size

  def __iter__(self):
    return self

  def __next__(self):
    """
    Generate the next batch of parameters for the Rosenbrock function.

    Returns:
        tuple: A tuple containing a numpy array of shape (batch_size, 2)
               where each row is [a, b], and None.
    """
    batch = jnp.array([[self.a, self.b, self.c]] * self.batch_size)
    return batch, None

def get_ackley_model_and_datagen(dim=3, a=20.0, b=0.2, c=2*jnp.pi):
  return EnhancedSequential([AckleyBlock(dimension=dim),]), AckleyIterator(a, b, c)

class GoldsteinPriceBlock(Module):
  """Consider parameter (x, y) as input and apply the Goldstein-Price function."""
  dtype: Any = jnp.float32
  precision: Any = None

  param_init: Callable[[PRNGKey, tuple, Any], jnp.ndarray] = lecun_uniform

  @compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    inputs = jnp.asarray(inputs, self.dtype)
    params = self.param("kernel", self.param_init(), (1, 2))
    params = jnp.asarray(params, self.dtype)
    inputs = inputs.flatten()
    params = params.flatten()

    # Compute Goldstein-Price function
    a, b = inputs[0], inputs[1] # dummy inputs, for compatibility with the current pipeline
    x_param, y_param = params[0], params[1]

    term1 = (
            1
            + (x_param + y_param + 1) ** 2
            * (19 - 14 * x_param + 3 * x_param ** 2 - 14 * y_param + 6 * x_param * y_param + 3 * y_param ** 2)
    )
    term2 = (
            30
            + (2 * x_param - 3 * y_param) ** 2
            * (18 - 32 * x_param + 12 * x_param ** 2 + 48 * y_param - 36 * x_param * y_param + 27 * y_param ** 2)
    )
    y = term1 * term2
    return y

def get_goldsteinprice_model_and_datagen():
  return EnhancedSequential([GoldsteinPriceBlock(), ]), RosenbrockIterator()