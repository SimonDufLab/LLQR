import abc
from typing import Tuple
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


class BlockStructures(abc.ABC):

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init):
    """The abstract base class for all block structures.

    Attributes:
      block_structure_init: Initialization value for all blocks. Currently, only support 'identity'
    """
    if block_structure_init == 'identity':
      self._init_blocks = self.identity_block_init
    self.blocks = self._make_blocks(network_params, layer_names)

  def update_blocks(self, new_blocks):
    self.blocks.update(new_blocks)

  @abc.abstractmethod
  def identity_block_init(self, shape:Tuple[int, ...]) -> jnp.ndarray:
    """Initialize the blocks so that the matrix product function at initialization is equivalent to
     an identity operation"""
    pass

  @abc.abstractmethod
  def _make_blocks(self,
                  network_params,
                  layer_names
                  ):
    """Create the preconditioner block for every layer. Can be used to impose structure to reduce memory usage.
       For example, by using a diagonal representation or a kronecker factorization"""
    pass

  def power_iteration(self): # TODO: Should be the same for all, leveraging matrix_product method
    """Performing the power iteration of every preconditioner block, to recover the biggest eigenvalue"""
    pass

  @abc.abstractmethod
  def matrix_product(self,
                     blocks,
                     vectors,
                     ):
    """Encode the matrix product definition w/r to the chosen structure.
       The vectors to multiply with will typically be the gradients,
       encoded in a dictionary with the same structure as weights"""


class DenseBlock(BlockStructures):
  """A dense block structure."""
  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               ):
    super().__init__(network_params, layer_names, block_structure_init)

  def identity_block_init(self, shape:Tuple[int, ...]) -> jnp.ndarray:
    return jnp.eye(*shape)

  def _make_blocks(self,
                  network_params,
                  layer_names
                  ):
    blocks = {layer_name: self._init_blocks((ravel_pytree(network_params[layer_name])[0].size,)*2)
                   for layer_name in layer_names}
    return blocks

  def matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      flat_product = blocks[layer_name].dot(flat_vector)
      product_dict[layer_name] = unravel_fn(flat_product)

    return product_dict
    # return jax.tree_map(jnp.dot, blocks, vectors)


class DiagonalBlock(BlockStructures):
  """A diagonal block structure."""
  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               ):
    super().__init__(network_params, layer_names, block_structure_init)

  def identity_block_init(self, shape:int) -> jnp.ndarray:
    return jnp.ones(shape)

  def _make_blocks(self,
                  network_params,
                  layer_names
                  ):
    blocks = {layer_name: self._init_blocks(ravel_pytree(network_params[layer_name])[0].size)
                   for layer_name in layer_names}
    return blocks

  def matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      flat_product = blocks[layer_name]*flat_vector
      product_dict[layer_name] = unravel_fn(flat_product)

    return product_dict