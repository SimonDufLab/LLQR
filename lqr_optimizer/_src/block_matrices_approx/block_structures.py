import abc
from typing import Tuple, Optional, Any
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.traverse_util import flatten_dict, unflatten_dict

def ema_update(old, new, decay):
  return old * decay + new * (1 - decay)


class BlockStructures(abc.ABC):

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank = None,
               identity_scale = 1.0, # Scaling factor of identity init when training preconditioner
               ):
    """The abstract base class for all block structures.

    Attributes:
      block_structure_init: Initialization value for all blocks. Currently, only support 'identity'
    """
    self.rank = rank
    self._identity_scale = identity_scale
    if block_structure_init == 'identity':
      self._init_blocks = self.identity_block_init
    self.blocks = self._make_blocks(network_params, layer_names, initialization=True)

    self.reinit_blocks = jax.jit(Partial(self._make_blocks, network_params, layer_names))

  def update_blocks(self, new_blocks, ema_decay=0):
    _new_blocks = jax.tree_map(Partial(ema_update, decay=ema_decay), self.blocks, new_blocks)
    self.blocks.update(_new_blocks)

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
                  layer_names,
                  initialization = False,
                  ):
    """Create the preconditioner block for every layer. Can be used to impose structure to reduce memory usage.
       For example, by using a diagonal representation or a kronecker factorization"""
    pass

  def power_iteration(self): # TODO: Should be the same for all, leveraging matrix_product method
    """Performing the power iteration of every preconditioner block, to recover the biggest eigenvalue"""
    pass

  @abc.abstractmethod
  def train_matrix_product(self,
                     blocks,
                     vectors,
                     ):
    """Encode the matrix product definition w/r to the chosen structure.
       The vectors to multiply with will typically be the gradients,
       encoded in a dictionary with the same structure as weights"""

    pass

  def matrix_product(self,
                     blocks,
                     vectors,
                     ):
    """
        Usually, same as train_matrix_product. Different by blocks using a memory.
        Cancel out scaling_factor for init used during training if any.
    """
    updated_grad = jax.tree_map(lambda v : v/self._identity_scale, self.train_matrix_product(blocks, vectors))
    return updated_grad
    # return self.train_matrix_product(blocks, vectors)


class DenseBlock(BlockStructures):
  """A dense block structure."""
  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape:Tuple[int, ...]) -> jnp.ndarray:
    return self._identity_scale * jnp.eye(*shape)

  def _make_blocks(self,
                  network_params,
                  layer_names,
                  initialization=False,
                  ):
    blocks = {layer_name: self._init_blocks((ravel_pytree(network_params[layer_name])[0].size,)*2)
                   for layer_name in layer_names}
    return blocks

  def train_matrix_product(self, blocks, vectors):
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
               rank,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape:int) -> jnp.ndarray:
    return self._identity_scale * jnp.ones(shape)

  def _make_blocks(self,
                  network_params,
                  layer_names,
                  initialization=False,
                  ):
    blocks = {layer_name: self._init_blocks(ravel_pytree(network_params[layer_name])[0].size)
                   for layer_name in layer_names}
    return blocks

  def train_matrix_product(self, blocks, vectors):
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
               rank,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape) -> jnp.ndarray:
    return self._identity_scale * jnp.ones((1,))

  def _make_blocks(self,
                  network_params,
                  layer_names,
                  initialization=False,
                  ):
    blocks = {layer_name: self._init_blocks(None)
                   for layer_name in layer_names}
    return blocks

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      flat_product = blocks[layer_name]*flat_vector
      product_dict[layer_name] = unravel_fn(flat_product)

    return product_dict

KERNEL_KEYS = ("kernel", "embedding", "pos_embedding")
BIAS_KEYS = ("bias", "scale")
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

  def __init__(self, network_params, layer_names, block_structure_init, rank, identity_scale=1.0):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, dim: int) -> jnp.ndarray:
    """Used for initializing Kronecker factors (as an identity matrix)."""
    return self._identity_scale * jnp.eye(dim)

  def identity_diag_init(self, dim: int) -> jnp.ndarray:
    """Used for initializing diagonal blocks (as a vector of ones)."""
    return self._identity_scale * jnp.ones((dim,))

  def _make_blocks(self, network_params, layer_names, initialization=False):
    blocks = {}
    for layer_name in layer_names:
      flat_params = flatten_dict(network_params[layer_name])
      layer_blocks = {}
      # Process kernel using Kronecker factorization
      for key in flat_params.keys():
        if any(k in key for k in KERNEL_KEYS):
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
            # param_shape = kernel.shape
            # Reshape the kernel to 2D: (k_h*k_w*cin, cout)
            # reshaped_kernel = kernel.reshape((k_h * k_w * cin, cout))
            factor_A = self.identity_block_init(k_h * k_w * cin)
            factor_B = self.identity_block_init(cout)
            layer_blocks[key] = (factor_A, factor_B)
          else:
            raise ValueError(f"Kernel shape {kernel.shape} not supported for layer {layer_name}")
        # Process bias using a diagonal approximation
        if any(k in key for k in BIAS_KEYS):
          bias = flat_params[key]
          if not hasattr(bias, "shape"):
            raise ValueError(f"Bias for layer {layer_name} does not have a shape attribute")
          flat_bias, unravel_fn_bias = ravel_pytree(bias)
          diag = self.identity_diag_init(flat_bias.shape[0])
          layer_blocks[key] = diag
      blocks[layer_name] = layer_blocks
    return blocks

  def train_matrix_product(self, blocks, vectors):
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
        if any(k in component for k in KERNEL_KEYS):
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
          # product_matrix = factor_A @ vector_matrix @ factor_B.T
          product_matrix = jnp.einsum('mk,kn,rn->mr', factor_A, vector_matrix, factor_B)
          layer_product[component] = product_matrix.reshape(vm_shape)
        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          # Elementwise product for the diagonal approximation
          flat_product = diag * vec_component
          layer_product[component] = flat_product
        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")
      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict

  # @staticmethod
  # def prepare_vectors(vectors):
  #   kernel_shapes = {}
  #   prepared_vectors = {}
  #   for layer_name, layer_vectors in vectors.items():
  #     prepared_layer_vectors = {}
  #     for component, vec_component in flatten_dict(layer_vectors).items():
  #       if 'kernel' in component:
  #         # Reshape the gradient vector according to the component type
  #         vector_matrix = vec_component
  #         vm_shape = vector_matrix.shape
  #         if len(vm_shape) == 4:
  #           k_h, k_w, cin, cout = vm_shape
  #           vector_matrix = vector_matrix.reshape((k_h * k_w * cin, cout))
  #         prepared_layer_vectors[component] = vector_matrix
  #         kernel_shapes[layer_name] = {component: vm_shape}
  #       elif 'bias' in component or 'scale' in component:
  #         prepared_layer_vectors[component] = vec_component
  #       else:
  #         raise ValueError(f"Unknown block type for {component} in layer {layer_name}")
  #     prepared_vectors[layer_name] = prepared_layer_vectors
  #   return prepared_vectors, kernel_shapes

  # @staticmethod
  # def matrix_product_for_scan(blocks, prepared_vectors, kernel_shapes):
  #   """
  #   For each layer, perform the appropriate matrix–vector product for each parameter component.
  #
  #   - For kernel (dense or conv):
  #       Compute vec(A @ X @ B^T), where X is the pre-reshaped gradient.
  #       Then reshape back to the original kernel shape from kernel_shapes.
  #   - For bias/scale:
  #       Compute elementwise multiplication with the diagonal block.
  #   """
  #   product_dict = {}
  #   for layer_name, layer_vectors in prepared_vectors.items():
  #     layer_blocks = blocks[layer_name]
  #     layer_product = {}
  #
  #     for component, vec_component in layer_vectors.items():
  #       if "kernel" in component:
  #             factor_A, factor_B = layer_blocks[component]
  #             vector_matrix = vec_component  # already 2D from prepare_vectors
  #
  #             # Apply Kronecker product approximation
  #             product_matrix = jnp.einsum(
  #                 "mk,kn,rn->mr", factor_A, vector_matrix, factor_B
  #             )
  #
  #             # Reshape back to the original 4D kernel shape
  #             vm_shape = kernel_shapes[layer_name][component]
  #             layer_product[component] = product_matrix.reshape(vm_shape)
  #
  #         # else: # Always called after prepare_vectors, so check is done
  #       elif 'bias' in component or 'scale' in component:
  #             diag = layer_blocks[component]
  #             layer_product[component] = diag * vec_component
  #     product_dict[layer_name] = unflatten_dict(layer_product)
  #
  #   return product_dict

class DiagKroneckerBlock(KroneckerBlock):
  """
  Diagonal Kronecker-factored block structure, in the spirit of AdaFisher.

  For kernel parameters:
    - We approximate the Kronecker factors A and B by their diagonals:
        A ≈ Diag(a),  B ≈ Diag(b)
      so that the action on a kernel matrix X is:
        A @ X @ B^T  ≈  Diag(a) @ X @ Diag(b)
                     =  a[:, None] * X * b[None, :]

    - Dense kernels (2D, shape (m, n)):
        store (a ∈ R^m, b ∈ R^n)

    - Conv kernels (4D, shape (k_h, k_w, cin, cout)):
        reshape to (k_h*k_w*cin, cout) for the Kronecker structure, and store:
        a ∈ R^{k_h*k_w*cin}, b ∈ R^{cout}

  Bias parameters keep the same diagonal approximation as in KroneckerBlock.
  """

  def __init__(self, network_params, layer_names, block_structure_init, rank, identity_scale=1.0):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def _make_blocks(self, network_params, layer_names, initialization=False):
    """
    For kernels, store only the diagonal of the Kronecker factors:

      - Dense: (diag_A, diag_B) with shapes (m,), (n,)
      - Conv:  (diag_A, diag_B) with shapes (k_h*k_w*cin,), (cout,)

    Bias logic is unchanged: a single diagonal vector.
    """
    blocks = {}
    for layer_name in layer_names:
      flat_params = flatten_dict(network_params[layer_name])
      layer_blocks = {}
      # Process kernel using *diagonal* Kronecker factorization
      for key in flat_params.keys():
        if any(k in key for k in KERNEL_KEYS):
          kernel = flat_params[key]
          if not hasattr(kernel, "shape"):
            raise ValueError(f"Kernel for layer {layer_name} does not have a shape attribute")
          if len(kernel.shape) == 2:
            # Dense layer kernel: shape (m, n)
            m, n = kernel.shape
            diag_A = self.identity_diag_init(m)  # a ∈ R^m
            diag_B = self.identity_diag_init(n)  # b ∈ R^n
            layer_blocks[key] = (diag_A, diag_B)
          elif len(kernel.shape) == 4:
            # Convolution layer kernel: shape (k_h, k_w, in_channels, out_channels)
            k_h, k_w, cin, cout = kernel.shape
            M = k_h * k_w * cin
            N = cout
            diag_A = self.identity_diag_init(M)  # a ∈ R^{k_h*k_w*cin}
            diag_B = self.identity_diag_init(N)  # b ∈ R^{cout}
            layer_blocks[key] = (diag_A, diag_B)
          else:
            raise ValueError(f"Kernel shape {kernel.shape} not supported for layer {layer_name}")

        # Process bias using a diagonal approximation (unchanged)
        if any(k in key for k in BIAS_KEYS):
          bias = flat_params[key]
          if not hasattr(bias, "shape"):
            raise ValueError(f"Bias for layer {layer_name} does not have a shape attribute")
          flat_bias, unravel_fn_bias = ravel_pytree(bias)
          diag = self.identity_diag_init(flat_bias.shape[0])
          layer_blocks[key] = diag

      blocks[layer_name] = layer_blocks
    return blocks

  def train_matrix_product(self, blocks, vectors):
    """
    Apply the diagonal Kronecker block to 'vectors'.

    For kernel params:
      Dense (m, n):
        X_out[i, j] = diag_A[i] * X[i, j] * diag_B[j]
        implemented as:
          jnp.einsum('i,ij,j->ij', diag_A, X, diag_B)

      Conv (k_h, k_w, cin, cout):
        reshape X -> (M, N) with M = k_h*k_w*cin, N = cout
        X_out_2d[i, j] = diag_A[i] * X_2d[i, j] * diag_B[j]
        implemented as:
          jnp.einsum('i,ij,j->ij', diag_A, X_2d, diag_B)
        then reshape back to original 4D shape.

    Bias logic remains the same: elementwise multiplication by the diagonal.
    """
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          diag_A, diag_B = layer_blocks[component]
          vector_matrix = vec_component
          vm_shape = vector_matrix.shape

          if len(vm_shape) == 2:
            # Dense kernel: (m, n)
            m, n = vm_shape
            if diag_A.shape[0] != m or diag_B.shape[0] != n:
              raise ValueError(
                f"Diagonal Kronecker shapes {diag_A.shape}, {diag_B.shape} "
                f"do not match dense kernel shape {vm_shape} for {component}"
              )
            # X_out[i, j] = diag_A[i] * X[i, j] * diag_B[j]
            product_matrix = jnp.einsum('i,ij,j->ij', diag_A, vector_matrix, diag_B)
            layer_product[component] = product_matrix

          elif len(vm_shape) == 4:
            # Conv kernel: (k_h, k_w, cin, cout)
            k_h, k_w, cin, cout = vm_shape
            M = k_h * k_w * cin
            N = cout
            if diag_A.shape[0] != M or diag_B.shape[0] != N:
              raise ValueError(
                f"Diagonal Kronecker shapes {diag_A.shape}, {diag_B.shape} "
                f"do not match conv kernel flattened shape {(M, N)} for {component}"
              )
            vector_2d = vector_matrix.reshape(M, N)
            # X_out_2d[i, j] = diag_A[i] * X_2d[i, j] * diag_B[j]
            product_2d = jnp.einsum('i,ij,j->ij', diag_A, vector_2d, diag_B)
            layer_product[component] = product_2d.reshape(vm_shape)

          else:
            raise ValueError(
              f"Unsupported kernel tensor rank {len(vm_shape)} for component {component} "
              f"in layer {layer_name}"
            )

        elif any(k in component for k in BIAS_KEYS):
          # Bias: elementwise product with diagonal (unchanged)
          diag = layer_blocks[component]
          flat_product = diag * vec_component
          layer_product[component] = flat_product

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict

##################################################################
## Block Structures inspired by Sherman-Morrison-Woodbury formula
##################################################################
class LowRankBlockMemory(BlockStructures):
  """
  A per-layer block structure with two components:
    1) A diagonal part (same as DiagonalBlock): shape (d,)
    2) A scalar part (same as ScalarBlock): shape (1,)

  Stored per layer as:
    blocks[layer_name] = {
      "diag":   <jnp.ndarray shape (d,)>,
      "scalar": <jnp.ndarray shape (1,)>,
    }

  Notes:
    - The scalar init can be configured to start at 1 or 0.
    - matrix_product is intentionally left unimplemented (pass).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank,
               identity_scale=1.0,
               ):

    assert isinstance(rank, int), "rank must be an int"
    self._key = jax.random.PRNGKey(42)
    self._memory = {l_name: {"diag": [], "scalar": 1.0} for l_name in layer_names}
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def get_memory(self):
    return {k:val['diag'] for k,val in self._memory.items()}

  # update rule now need to account for memory
  def update_blocks(self, new_blocks, ema_decay=0):
    """
    Instead of updating `self.blocks`, push the incoming `new_blocks` into
    `self.memory` with a FIFO replacement policy.

    - `self.memory` is created on first call.
    - For each layer, we maintain two FIFO lists: "diag" and "scalar".
    - The length of each list is bounded by `self.rank`.
    - `new_blocks[layer]` may contain only one of {"diag", "scalar"}; we only
      append what is present.
    """
    if not hasattr(self, "rank"):
      raise AttributeError("LowRankBlock must define `self.rank` to bound the FIFO memory.")

    for layer_name, layer_new in new_blocks.items():
      # Ensure per-layer containers exist
      if layer_name not in self._memory:
        self._memory[layer_name] = {"diag": [], "scalar": 1.0}

      # `layer_new` is expected to be a dict that contains either "diag" or "scalar" (or both)
      if "scalar" in layer_new:
        self._memory[layer_name]["scalar"] = (layer_new["scalar"])

      if "diag" in layer_new:
        self._memory[layer_name]["diag"].append(layer_new["diag"])
        # FIFO truncation
        if len(self._memory[layer_name]["diag"]) > self.rank:
          self._memory[layer_name]["diag"].pop(0)
    self.blocks = self.reinit_blocks()

  # --- Init helpers ---
  def diag_block_init(self, d: int) -> jnp.ndarray:
    """Diagonal init: random small vectors."""
    consumable, self._key = jax.random.split(self._key)
    return self._identity_scale * jax.random.normal(consumable, (d,)) / jnp.sqrt(d)

  def identity_block_init(self, shape: int) -> jnp.ndarray:
    """Scalar init: either 1 or 0, as shape (1,)."""
    return self._identity_scale * jnp.ones((1,))

  # --- Block construction ---
  def _make_blocks(self,
                   network_params,
                   layer_names,
                   initialization=False,):
    blocks = {}
    for layer_name in layer_names:
      d = ravel_pytree(network_params[layer_name])[0].size
      ## Scaling identity init when training begin is unstable, removing
      # if initialization:
      #   blocks[layer_name] = {
      #     "scalar": self.scalar_block_init(),  # identical to ScalarBlock (init selectable)
      #   }
      # else:
      blocks[layer_name] = {
        "diag": self.diag_block_init(d),  # identical to DiagonalBlock
      }
    return blocks

  # --- Product ---
  def matrix_product(self, blocks, vectors):
    """Ignore blocks, take memory only"""
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      # flat_vector = self._memory[layer_name]["scalar"] * flat_vector
      for u in self._memory[layer_name]["diag"]:
        flat_vector += u * jnp.dot(u, flat_vector)
      product_dict[layer_name] = unravel_fn(flat_vector)

    return jax.tree_map(lambda v: v/self._identity_scale, product_dict)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      # flat_vector = self._memory[layer_name]["scalar"] * flat_vector
      # if "scalar" in blocks[layer_name]:
      #   flat_vector = blocks[layer_name]["scalar"] * flat_vector
      start = max(0, len(self._memory[layer_name]["diag"])-self.rank + 1)
      for u in self._memory[layer_name]["diag"][start::]:
        flat_vector += u * jnp.dot(u, flat_vector)
      if "diag" in blocks[layer_name]:
        flat_vector += blocks[layer_name]["diag"] * jnp.dot(blocks[layer_name]["diag"], flat_vector)
      product_dict[layer_name] = unravel_fn(flat_vector)

    return product_dict

class LowRankBlockMemoryAsym(LowRankBlockMemory):
  """
  Asymmetric version of LowRankBlockMemory
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  # --- Block construction ---
  def _make_blocks(self,
                   network_params,
                   layer_names,
                   initialization=False,):
    blocks = {}
    for layer_name in layer_names:
      d = ravel_pytree(network_params[layer_name])[0].size
      blocks[layer_name] = {
        "diag": (self.diag_block_init(d), self.diag_block_init(d)),  # identical to DiagonalBlock
      }
    return blocks

  # --- Product ---
  def matrix_product(self, blocks, vectors):
    """Ignore blocks, take memory only"""
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      for u, v in self._memory[layer_name]["diag"]:
        flat_vector += u * jnp.dot(v, flat_vector)
      product_dict[layer_name] = unravel_fn(flat_vector)

    return jax.tree_map(lambda vv: vv/self._identity_scale, product_dict)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, block_vector in vectors.items():
      flat_vector, unravel_fn = ravel_pytree(block_vector)
      start = max(0, len(self._memory[layer_name]["diag"])-self.rank + 1)
      for u, v in self._memory[layer_name]["diag"][start::]:
        flat_vector += u * jnp.dot(u, flat_vector)
      if "diag" in blocks[layer_name]:
        flat_vector += blocks[layer_name]["diag"][0] * jnp.dot(blocks[layer_name]["diag"][1], flat_vector)
      product_dict[layer_name] = unravel_fn(flat_vector)

    return product_dict


@jax.jit
def sherman_morrison_rank1(
    inv_matrix: jnp.ndarray,
    vectors: Tuple[jnp.ndarray, jnp.ndarray],
    eps: float = 1e-8,
) -> jnp.ndarray:
    """
    Compute (A + u v^T)^(-1) from A^(-1) via Sherman–Morrison, with epsilon-stabilized denom.

    Supports:
      - inv_matrix shape (n, n): dense A^{-1}
      - inv_matrix shape (n,): interpreted as diagonal A^{-1} (i.e., diag(inv_matrix))

    Args:
        inv_matrix: A_inv, shape (n,n) or (n,)
        vectors: (u, v) with shape (n,) or (n,1)
        eps: small stabilization constant for the denominator

    Returns:
        Updated inverse (dense), shape (n, n)
    """
    Ainv = inv_matrix
    u, v = vectors
    u = jnp.reshape(u, (-1,))
    v = jnp.reshape(v, (-1,))

    # Promote diagonal representation to dense (note: rank-1 update yields dense inverse in general)
    # This creates a dense matrix to capture second order moment and bias and norm param # TODO is it better to stick with vector approximation?
    # Ainv = jnp.diag(inv_matrix) if inv_matrix.ndim == 1 else inv_matrix

    Au = Ainv @ u             # (n,)
    vA = v @ Ainv             # (n,)
    denom = 1.0 + (v @ Au)    # ()

    # Epsilon-stabilize denom to avoid division by ~0
    sign = jnp.where(denom >= 0, 1.0, -1.0)
    denom_safe = jnp.where(jnp.abs(denom) < eps, sign * eps, denom)

    return Ainv - jnp.outer(Au, vA) / denom_safe


def upgrade_1d_leaves_to_diag(pytree: Any) -> Any:
  """
  Traverse a pytree and replace every 1D array leaf (shape (n,)) with jnp.diag(leaf) (shape (n, n)).
  Non-array leaves and arrays with ndim != 1 are returned unchanged.
  """

  def f(leaf: Any) -> Any:
    if isinstance(leaf, jnp.ndarray) and leaf.ndim == 1:
      return jnp.diag(leaf)
    return leaf

  return jax.tree_map(f, pytree)


class Sym_SWM_KFAC(KroneckerBlock):
  """Shermann-Woodbury-Morisson KFAC structure
  Kronecker structure for the preconditioner, but with rank-one update"""
  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               ):

    self._key = jax.random.PRNGKey(42)
    self._identity_scale = identity_scale
    self._memory = upgrade_1d_leaves_to_diag(super()._make_blocks(network_params, layer_names, initialization=True))  # Memory initialized to identity
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def get_memory(self):
    return self._memory

  def matrix_product(self, blocks, vectors): # Only use memory outside of training.
    # return super().train_matrix_product(self._memory, vectors)
    """Same as kfac, but ignore blocks, use memory instead"""
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = self._memory[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          factor_A, factor_B = layer_blocks[component]
          # Reshape the gradient vector according to the component type
          vector_matrix = vec_component
          vm_shape = vector_matrix.shape
          if len(vm_shape) == 4:
            k_h, k_w, cin, cout = vm_shape
            vector_matrix = vector_matrix.reshape((k_h * k_w * cin, cout))
          product_matrix = jnp.einsum('mk,kn,rn->mr', factor_A, vector_matrix, factor_B)
          layer_product[component] = product_matrix.reshape(vm_shape)
        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          # rank-1 update
          flat_product = diag @ vec_component
          layer_product[component] = flat_product
        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")
      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict

  def diag_block_init(self, d: int) -> jnp.ndarray:
    """Diagonal init: random small vectors."""
    consumable, self._key = jax.random.split(self._key)
    return self._identity_scale * jax.random.normal(consumable, (d,)) / jnp.sqrt(d)

  def _make_blocks(self,
                   network_params,
                   layer_names,
                   initialization=False,):
    def pytee_init(leaf):
      d = leaf.shape[0]
      return self.diag_block_init(d)
    return jax.tree_map(pytee_init, self._memory)

  def _shermann_morisson_update(self, u):
    """Symmetric version"""
    return jax.tree_map(lambda A, x: sherman_morrison_rank1(A, (x, x)), self._memory, u)

  def train_matrix_product(self, blocks, vectors):
    blocks = self._shermann_morisson_update(blocks)
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          factor_A, factor_B = layer_blocks[component]
          # Reshape the gradient vector according to the component type
          vector_matrix = vec_component
          vm_shape = vector_matrix.shape
          if len(vm_shape) == 4:
            k_h, k_w, cin, cout = vm_shape
            vector_matrix = vector_matrix.reshape((k_h * k_w * cin, cout))
          product_matrix = jnp.einsum('mk,kn,rn->mr', factor_A, vector_matrix, factor_B)
          layer_product[component] = product_matrix.reshape(vm_shape)
        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          # Now a matrix with rank-1 update, not a vector anymore
          flat_product = diag @ vec_component
          layer_product[component] = flat_product
        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")
      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict

  def update_blocks(self, new_blocks, ema_decay=0):
    self._memory = self._shermann_morisson_update(new_blocks)
    self.blocks = self.reinit_blocks()


class Asym_SWM_KFAC(Sym_SWM_KFAC):
  """Shermann-Woodbury-Morisson KFAC structure - -asymmetric version
  Kronecker structure for the preconditioner, but with rank-one update"""
  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def _make_blocks(self,
                   network_params,
                   layer_names,
                   initialization=False, ):
    def pytee_init(leaf):
      d = leaf.shape[0]
      return self.diag_block_init(d), self.diag_block_init(d)

    return jax.tree_map(pytee_init, self._memory)

  def _shermann_morisson_update(self, uv):
    """asymmetric version"""
    # print(jax.tree_map(jnp.shape, uv))
    # is_tuple_leaf = lambda x: isinstance(x, tuple)
    # u = jax.tree_map(lambda t: t[0], uv, is_leaf=is_tuple_leaf)
    # v = jax.tree_map(lambda t: t[1], uv, is_leaf=is_tuple_leaf)
    #
    # print(jax.tree_map(jnp.shape, u))
    # print(jax.tree_map(jnp.shape, v))
    def _select_pair(t, i: int):
      """Recursively pick element i from nested (len-2) tuple pairs."""
      if not (isinstance(t, tuple) and len(t) == 2):
        raise TypeError(f"Expected a length-2 tuple, got {type(t)}")
      a, b = t

      # Base case: (array, array)  -> pick one
      if not (isinstance(a, tuple) and len(a) == 2) and not (isinstance(b, tuple) and len(b) == 2):
        return t[i]

      # Nested case: ((..., ...), (..., ...)) -> recurse, preserving structure
      return (_select_pair(a, i), _select_pair(b, i))

    is_pair = lambda x: isinstance(x, tuple) and len(x) == 2

    u = jax.tree_map(lambda t: _select_pair(t, 0), uv, is_leaf=is_pair)
    v = jax.tree_map(lambda t: _select_pair(t, 1), uv, is_leaf=is_pair)
    return jax.tree_map(lambda A, x, y: sherman_morrison_rank1(A, (x, y)), self._memory, u, v)