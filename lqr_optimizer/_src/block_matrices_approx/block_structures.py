import abc
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.traverse_util import flatten_dict, unflatten_dict


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

  def clip_blocks(self, min_for_block=None, max_for_block=None):
    block_clip_fn = Partial(jnp.clip, min=min_for_block, max=max_for_block)
    self.blocks = jax.tree_map(block_clip_fn, self.blocks)

  @abc.abstractmethod
  def identity_block_init(self, shape:Optional[Tuple[int, ...]]) -> jnp.ndarray:
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

class ScalarBlock(BlockStructures):
  """A scalar per layer, to reduce memory usage"""
  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               ):
    super().__init__(network_params, layer_names, block_structure_init)

  def identity_block_init(self, shape) -> jnp.ndarray:
    return jnp.ones((1,))

  def _make_blocks(self,
                  network_params,
                  layer_names
                  ):
    blocks = {layer_name: self._init_blocks(None)
                   for layer_name in layer_names}
    return blocks

  def matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      flat_product = blocks[layer_name]*flat_vector
      product_dict[layer_name] = unravel_fn(flat_product)

    return product_dict


class KroneckerBlock(BlockStructures):
  """
  A block structure that handles both kernel and bias parameters.

  For the kernel:
    - Dense (2D) kernels are approximated with a Kronecker product:
         vec(B @ X @ A^T)
      where A and B are identity matrices initialized with sizes matching the kernel dimensions.
    - Convolution kernels (4D with shape (k_h, k_w, in_channels, out_channels))
      are reshaped to a 2D matrix of shape (k_h * k_w * in_channels, out_channels) and factorized similarly.

  For the bias (if present):
    - A diagonal approximation is applied (elementwise multiplication with a diagonal vector).
  """

  def __init__(self, network_params, layer_names, block_structure_init):
    super().__init__(network_params, layer_names, block_structure_init)

  def identity_block_init(self, dim: int) -> jnp.ndarray:
    """Used for initializing Kronecker factors (as an identity matrix)."""
    return jnp.eye(dim)

  def identity_diag_init(self, dim: int) -> jnp.ndarray:
    """Used for initializing diagonal blocks (as a vector of ones)."""
    return jnp.ones((dim,))

  def _make_blocks(self, network_params, layer_names):
    blocks = {}
    for layer_name in layer_names:
      flat_params = flatten_dict(network_params[layer_name])
      layer_blocks = {}
      # Process kernel using Kronecker factorization
      for key in flat_params.keys():
        if "kernel" in key:
          kernel = flat_params[key]
          if not hasattr(kernel, "shape"):
            raise ValueError(f"Kernel for layer {layer_name} does not have a shape attribute")
          if len(kernel.shape) == 2:
            # Dense layer kernel: shape (m, n)
            param_shape = kernel.shape
            factor_A = self.identity_block_init(param_shape[0])
            factor_B = self.identity_block_init(param_shape[1])
            layer_blocks[key] = (factor_A, factor_B)
          elif len(kernel.shape) == 4:
            # Convolution layer kernel: shape (k_h, k_w, in_channels, out_channels)
            k_h, k_w, cin, cout = kernel.shape
            param_shape = kernel.shape
            # Reshape the kernel to 2D: (k_h*k_w*cin, cout)
            reshaped_kernel = kernel.reshape((k_h * k_w * cin, cout))
            factor_A = self.identity_block_init(k_h * k_w * cin)
            factor_B = self.identity_block_init(cout)
            layer_blocks[key] = (factor_A, factor_B)
          else:
            raise ValueError(f"Kernel shape {kernel.shape} not supported for layer {layer_name}")
        # Process bias using a diagonal approximation
        if "bias" in key or 'scale' in key:
          bias = flat_params[key]
          if not hasattr(bias, "shape"):
            raise ValueError(f"Bias for layer {layer_name} does not have a shape attribute")
          flat_bias, unravel_fn_bias = ravel_pytree(bias)
          diag = self.identity_diag_init(flat_bias.shape[0])
          layer_blocks[key] = diag
      blocks[layer_name] = layer_blocks
    return blocks

  def matrix_product(self, blocks, vectors):
    """
    For each layer, perform the appropriate matrix–vector product for each parameter component.

    For kernel (dense or conv):
      Compute vec(B @ X @ A^T), where X is the gradient reshaped to the original parameter shape.
    For bias:
      Compute elementwise multiplication using the diagonal vector.
    """
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():
        if 'kernel' in component:
          factor_A, factor_B = layer_blocks[component]
          # Reshape the gradient vector according to the component type
          vector_matrix = vec_component
          vm_shape = vector_matrix.shape
          if len(vm_shape) == 4:
            k_h, k_w, cin, cout = vm_shape
            vector_matrix = vector_matrix.reshape((k_h * k_w * cin, cout))
          # Apply the Kronecker product approximation
          # product_matrix = factor_B @ vector_matrix.T @ factor_A.T
          # layer_product[component] = product_matrix.T.reshape(vm_shape)
          product_matrix = factor_A @ vector_matrix @ factor_B.T
          layer_product[component] = product_matrix.reshape(vm_shape)
        elif 'bias' in component or 'scale' in component:
          diag = layer_blocks[component]
          # Elementwise product for the diagonal approximation
          flat_product = diag * vec_component
          layer_product[component] = flat_product
        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")
      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict

