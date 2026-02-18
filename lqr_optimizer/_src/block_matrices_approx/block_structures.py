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
## KFAC block structures enforcing symmetry
##################################################################
# -----------------------------
# Dense symmetry helpers
# -----------------------------
def _lower_triangle_param_init(dim: int, scale: float, *, kind: str) -> jnp.ndarray:
  """
  Initialize a dense (dim, dim) lower-triangular parameter matrix.
  The upper-triangular part is zeroed (non-parameter region).

  kind:
    - "mirror": want P = T + T^T - diag(T) to equal scale*I at init -> set diag(T)=scale
    - "psd":    want P = L L^T to equal scale*I at init -> set diag(L)=sqrt(scale)
    - "sds":    want S ~ I initially -> set diag(S)=1
  """
  if kind == "mirror":
    diag = jnp.full((dim,), scale, dtype=jnp.float32)
  elif kind == "psd":
    diag = jnp.full((dim,), jnp.sqrt(scale), dtype=jnp.float32)
  elif kind == "sds":
    diag = jnp.ones((dim,), dtype=jnp.float32)
  else:
    raise ValueError(f"Unknown kind: {kind}")

  M = jnp.zeros((dim, dim), dtype=jnp.float32)
  M = M.at[jnp.arange(dim), jnp.arange(dim)].set(diag)
  return jnp.tril(M)  # ensure upper is zero


@jax.jit
def _project_to_lower_triangle(M: jnp.ndarray) -> jnp.ndarray:
  """Keep only the lower-triangular part (including diag). Upper part is discarded."""
  return jnp.tril(M)


@jax.jit
def _apply_mirror_sym(T_lower: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
  """
  Apply P@X with P = T + T^T - diag(T), where T_lower stores only lower-tri params.
  Upper is derived and tied automatically.

  Efficient form: (T@X) + (T^T@X) - diag(T)*X
  """
  TX = T_lower @ X
  TtX = T_lower.T @ X
  diagT = jnp.diag(T_lower)[:, None]
  return TX + TtX - diagT * X


@jax.jit
def _apply_psd_sym(L_lower: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
  """
  Apply (L L^T) @ X without forming L L^T:
    tmp = L^T X
    out = L tmp
  """
  return L_lower @ (L_lower.T @ X)


@jax.jit
def _apply_sds_sym(S_lower: jnp.ndarray, D: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
  """
  Apply (S D S^T) @ X without forming S D S^T.
  S_lower is dense lower-tri, D is (n,).
  """

  StX = S_lower.T @ X
  DStX = D[:, None] * StX
  return S_lower @ DStX


@jax.jit
def _apply_sds_sym_norm_d(S_lower: jnp.ndarray, D: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
  """
  Normalize D
  Apply (S D S^T) @ X without forming S D S^T.
  S_lower is dense lower-tri, D is (n,).
  """
  scale = jnp.max(jnp.abs(D)) + 1e-8
  D_eff = D / scale

  StX = S_lower.T @ X
  DStX = D_eff[:, None] * StX
  return S_lower @ DStX


def _reshape_kernel_to_2d(vec_component: jnp.ndarray) -> Tuple[jnp.ndarray, Tuple[int, ...]]:
  """Dense/conv kernel reshape helper."""
  orig_shape = vec_component.shape
  if vec_component.ndim == 4:
    k_h, k_w, cin, cout = orig_shape
    return vec_component.reshape((k_h * k_w * cin, cout)), orig_shape
  elif vec_component.ndim == 2:
    return vec_component, orig_shape
  else:
    raise ValueError(f"Kernel shape {orig_shape} not supported")


# -----------------------------
# 1) Mirror-symmetric factors: P = T + T^T - diag(T)
# -----------------------------
class MirrorSymKroneckerBlock(KroneckerBlock):
  """
  Dense lower-triangular parameter T (stored as full (n,n) with upper forced to 0):
    P := T + T^T - diag(T)

  Symmetry tying:
    - Only T_lower is stored/updated.
    - Upper part is never a free parameter.
  """

  def identity_block_init(self, dim: int) -> jnp.ndarray:
    # Choose T=scale*I => P=scale*I
    return _lower_triangle_param_init(dim, self._identity_scale, kind="mirror")

  def update_blocks(self, new_blocks, ema_decay=0):
    """
    Override to enforce parameterization after updates:
      - apply EMA on the *parameter* (lower-tri part)
      - re-project to lower-tri to keep symmetry tied
    """
    def upd(old, new):
      out = ema_update(old, new, decay=ema_decay)
      # enforce only-lower parameters
      if out.ndim == 2:
        out = _project_to_lower_triangle(out)
      return out

    _new_blocks = jax.tree_map(upd, self.blocks, new_blocks)
    self.blocks.update(_new_blocks)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():

        if any(k in component for k in KERNEL_KEYS):
          T_A, T_B = layer_blocks[component]  # each is (m,m) and (n,n), lower-tri parameter matrices
          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          # Apply A@X
          AX = _apply_mirror_sym(T_A, X2d)

          # Apply right multiply by B (symmetric): AX@B = (B@AX^T)^T
          out2d = _apply_mirror_sym(T_B, AX.T).T

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict


# -----------------------------
# 2) PSD symmetric factors: P = L L^T
# -----------------------------
class PSDSymKroneckerBlock(KroneckerBlock):
  """
  Dense lower-triangular parameter L (stored as full (n,n) with upper forced to 0):
    P := L L^T

  Symmetry + PSD by construction; upper tied automatically.
  """

  def identity_block_init(self, dim: int) -> jnp.ndarray:
    # Choose L=sqrt(scale)*I => P=scale*I
    return _lower_triangle_param_init(dim, self._identity_scale, kind="psd")

  def update_blocks(self, new_blocks, ema_decay=0):
    def upd(old, new):
      out = ema_update(old, new, decay=ema_decay)
      if out.ndim == 2:
        out = _project_to_lower_triangle(out)
      return out
    _new_blocks = jax.tree_map(upd, self.blocks, new_blocks)
    self.blocks.update(_new_blocks)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():

        if any(k in component for k in KERNEL_KEYS):
          L_A, L_B = layer_blocks[component]  # (m,m), (n,n), lower-tri params
          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          AX = _apply_psd_sym(L_A, X2d)
          out2d = _apply_psd_sym(L_B, AX.T).T

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict


# -----------------------------
# 3) Spectral-ish symmetric factors: P = S D S^T
# -----------------------------
class SDSSymKroneckerBlock(KroneckerBlock):
  """
  Dense lower-triangular parameter S and diagonal D:
    P := S D S^T

  - Symmetric always.
  - Indefinite allowed if D has negative entries.
  - Upper triangle tied automatically (S stored lower-tri only).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, dim: int):
    # S = I, D = scale * 1 => P = scale*I
    S = _lower_triangle_param_init(dim, self._identity_scale, kind="sds")  # diag=1
    D = self._identity_scale * jnp.ones((dim,), dtype=jnp.float32)
    return S, D

  def _make_blocks(self, network_params, layer_names, initialization=False):
    """
    Must override because each factor is a tuple (S, D) rather than a single matrix.
    """
    blocks = {}
    for layer_name in layer_names:
      flat_params = flatten_dict(network_params[layer_name])
      layer_blocks = {}

      for key in flat_params.keys():
        if any(k in key for k in KERNEL_KEYS):
          kernel = flat_params[key]
          if len(kernel.shape) == 2:
            m, n = kernel.shape
          elif len(kernel.shape) == 4:
            k_h, k_w, cin, cout = kernel.shape
            m, n = k_h * k_w * cin, cout
          else:
            raise ValueError(f"Kernel shape {kernel.shape} not supported for layer {layer_name}")

          SA = self.identity_block_init(m)  # (S_A, D_A)
          SB = self.identity_block_init(n)  # (S_B, D_B)
          layer_blocks[key] = (SA, SB)

        if any(k in key for k in BIAS_KEYS):
          bias = flat_params[key]
          flat_bias, _ = ravel_pytree(bias)
          layer_blocks[key] = self.identity_diag_init(flat_bias.shape[0])

      blocks[layer_name] = layer_blocks
    return blocks

  def update_blocks(self, new_blocks, ema_decay=0):
    """
    Enforce parameterization:
      - S is lower-tri only (project)
      - D stays vector; optional normalization is applied at apply time, not here
    """
    def upd(old, new):
      # old/new may be (S,D) tuples or arrays
      if isinstance(old, tuple):
        (S_old, D_old) = old
        (S_new, D_new) = new
        S = ema_update(S_old, S_new, decay=ema_decay)
        D = ema_update(D_old, D_new, decay=ema_decay)
        S = _project_to_lower_triangle(S)
        return (S, D)
      else:
        out = ema_update(old, new, decay=ema_decay)
        if out.ndim == 2:
          out = _project_to_lower_triangle(out)
        return out

    _new_blocks = jax.tree_map(upd, self.blocks, new_blocks)
    self.blocks.update(_new_blocks)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():

        if any(k in component for k in KERNEL_KEYS):
          (S_A, D_A), (S_B, D_B) = layer_blocks[component]
          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          AX = _apply_sds_sym(S_A, D_A, X2d)
          out2d = _apply_sds_sym(S_B, D_B, AX.T).T

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict


class NormalizedSDSSymKroneckerBlock(SDSSymKroneckerBlock):
  """
  Dense lower-triangular parameter S and diagonal D:
    P := S D S^T

  - Symmetric always.
  - Indefinite allowed if D has negative entries.
  - Upper triangle tied automatically (S stored lower-tri only).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               ):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}
      for component, vec_component in flatten_dict(layer_vectors).items():

        if any(k in component for k in KERNEL_KEYS):
          (S_A, D_A), (S_B, D_B) = layer_blocks[component]
          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          AX = _apply_sds_sym_norm_d(S_A, D_A, X2d)
          out2d = _apply_sds_sym_norm_d(S_B, D_B, AX.T).T

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict


##################################################################
## EKFAC Block Structures
##################################################################
class EKFACBlock(KroneckerBlock):
  """
  EKFAC-like Kronecker block structure.

  For each kernel (m x n):
    Store (Q_A, Q_G, inv_diag) where
      Q_A: (m, m) orthonormal-ish
      Q_G: (n, n) orthonormal-ish
      inv_diag: (m*n,) diagonal of the *inverse* in the Kronecker-eigenbasis

    Apply:
      X_hat = Q_A^T X Q_G
      vec_out_hat = inv_diag * vec(X_hat)
      out = Q_A out_hat Q_G^T

  Bias:
    unchanged diagonal vector.

  Notes:
    - This matches the EKFAC operator form (Q⊗Q) diag(.) (Q⊗Q)^T.
    - inv_diag is what you want if you "learn/apply the inverse directly".
      If you instead store s (not inverted), then you'd apply 1/(s + damping) here.
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape_or_dim):
    """
    Unlike KroneckerBlock.identity_block_init(dim)->matrix,
    EKFAC needs to know BOTH dims (m,n) to init inv_diag of length m*n.

    We'll accept either:
      - dim: int (fallback, used nowhere for kernels)
      - shape: tuple (m,n)
    """
    if isinstance(shape_or_dim, int):
      dim = shape_or_dim
      return self._identity_scale * jnp.eye(dim, dtype=jnp.float32)

    if len(shape_or_dim) != 2:
      raise ValueError(f"EKFAC identity_block_init expects (m,n), got {shape_or_dim}")
    m, n = shape_or_dim

    Q_A = jnp.eye(m, dtype=jnp.float32)
    Q_G = jnp.eye(n, dtype=jnp.float32)

    # We store inverse diagonal so that apply is identity at init:
    # X_hat = X, vec_out_hat = 1 * vec(X_hat), so inv_diag should be 1.
    inv_diag = jnp.ones((m * n,), dtype=jnp.float32)

    # Keep same scaling convention as the base class:
    # BlockStructures.matrix_product divides by self._identity_scale after train_matrix_product.
    # So to get identity overall, train_matrix_product should output identity_scale * X.
    # Easiest: scale inv_diag by identity_scale.
    inv_diag = self._identity_scale * inv_diag

    return (Q_A, Q_G, inv_diag)

  def _make_blocks(self, network_params, layer_names, initialization=False):
    blocks = {}
    for layer_name in layer_names:
      flat_params = flatten_dict(network_params[layer_name])
      layer_blocks = {}

      for key in flat_params.keys():
        if any(k in key for k in KERNEL_KEYS):
          kernel = flat_params[key]
          if not hasattr(kernel, "shape"):
            raise ValueError(f"Kernel for layer {layer_name} does not have a shape attribute")

          if len(kernel.shape) == 2:
            m, n = kernel.shape
          elif len(kernel.shape) == 4:
            k_h, k_w, cin, cout = kernel.shape
            m, n = k_h * k_w * cin, cout
          else:
            raise ValueError(f"Kernel shape {kernel.shape} not supported for layer {layer_name}")

          layer_blocks[key] = self.identity_block_init((m, n))

        if any(k in key for k in BIAS_KEYS):
          bias = flat_params[key]
          if not hasattr(bias, "shape"):
            raise ValueError(f"Bias for layer {layer_name} does not have a shape attribute")
          flat_bias, _ = ravel_pytree(bias)
          layer_blocks[key] = self.identity_diag_init(flat_bias.shape[0])

      blocks[layer_name] = layer_blocks
    return blocks

  @staticmethod
  def _ekfac_apply(Q_A: jnp.ndarray, Q_G: jnp.ndarray, inv_diag: jnp.ndarray, X2d: jnp.ndarray) -> jnp.ndarray:
    """
    Apply (Q_G ⊗ Q_A) diag(inv_diag) (Q_G ⊗ Q_A)^T to vec(X2d),
    returned as matrix shape (m,n).
    """
    # X_hat = Q_A^T X Q_G
    X_hat = (Q_A.T @ X2d) @ Q_G

    # Elementwise scaling in the Kronecker-eigenbasis
    X_hat_scaled = (inv_diag * X_hat.reshape(-1)).reshape(X2d.shape)

    # Back-transform
    return (Q_A @ X_hat_scaled) @ Q_G.T

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}

      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          Q_A, Q_G, inv_diag = layer_blocks[component]

          X = vec_component
          orig_shape = X.shape
          if X.ndim == 4:
            k_h, k_w, cin, cout = orig_shape
            X2d = X.reshape((k_h * k_w * cin, cout))
          elif X.ndim == 2:
            X2d = X
          else:
            raise ValueError(f"Kernel shape {orig_shape} not supported for layer {layer_name}")

          out2d = self._ekfac_apply(Q_A, Q_G, inv_diag, X2d)
          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)
    return product_dict


# --------------------------
# Normalization utilities
# --------------------------
def _normalize_diag(d: jnp.ndarray, mode: str, eps: float) -> jnp.ndarray:
  """
  Normalize a 1D vector d according to mode.
  Returns d_eff with the SAME shape.
  """
  if mode == "none":
    return d
  if mode == "maxabs":
    s = jnp.max(jnp.abs(d)) + eps
    return d / s
  if mode == "rms":
    s = jnp.sqrt(jnp.mean(d * d)) + eps
    return d / s
  if mode == "meanabs":
    s = jnp.mean(jnp.abs(d)) + eps
    return d / s
  if mode == "medianabs":
    s = jnp.median(jnp.abs(d)) + eps
    return d / s
  raise ValueError(f"Unknown normalization mode: {mode}")


class EKFACBlockNormalized(EKFACBlock):
  """
  Full EKFAC block with optional diagonal normalization at APPLY TIME.

  Stored per kernel:
    - QA: (m,m)
    - QG: (n,n)
    - inv_diag: (m*n,)  (can be indefinite)

  Apply:
    X_hat = QA^T X QG
    X_hat_scaled = diag_eff * vec(X_hat)  (diag_eff derived from inv_diag with normalization)
    out = QA X_hat_scaled QG^T

  Normalization rationale:
    - controls global gain drift while keeping anisotropy ("shape") of inv_diag
    - reduces LR sensitivity; prevents hidden effective LR schedules
    - stabilizes EMA+momentum updates of the diagonal in your LQR-trained setting

  Options:
    norm_mode in {"none","rms","maxabs","meanabs","medianabs","tanh"}
    - "tanh" performs soft clipping: d_eff = c * tanh(d/c)

  Optional calibration:
    - use_alpha=True: multiply normalized diag by a scalar alpha per kernel component.
      alpha is stored in the blocks and updated by your existing update mechanism.
      Initialize alpha so that overall init acts like identity.

  Identity init convention:
    Your BlockStructures.matrix_product divides by identity_scale after train_matrix_product.
    So we want train_matrix_product to output identity_scale * X at init.

    - If norm_mode="none": set inv_diag = identity_scale * 1
    - If norm_mode normalizes to unit scale: set alpha = identity_scale, and inv_diag = 1
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               *,
               norm_mode: str = "none",
               eps: float = 1e-12,
               tanh_c: float = 10.0,
               use_alpha: bool = False):
    self._norm_mode = norm_mode
    self._eps = float(eps)
    self._tanh_c = float(tanh_c)
    self._use_alpha = bool(use_alpha)
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape):
    if len(shape) != 2:
      raise ValueError(f"EKFACBlockNormalized.identity_block_init expects (m,n), got {shape}")
    m, n = shape
    QA = jnp.eye(m, dtype=jnp.float32)
    QG = jnp.eye(n, dtype=jnp.float32)

    if self._norm_mode == "none" and (not self._use_alpha):
      # Make train_matrix_product = identity_scale * X directly via inv_diag
      inv_diag = self._identity_scale * jnp.ones((m * n,), dtype=jnp.float32)
      return (QA, QG, inv_diag)

    # Otherwise, set diag "shape" to ones, and let alpha handle scale if requested.
    inv_diag = jnp.ones((m * n,), dtype=jnp.float32)

    if self._use_alpha:
      alpha = jnp.asarray(self._identity_scale, dtype=jnp.float32)
      return (QA, QG, inv_diag, alpha)

    # If no alpha, we can still try to preserve scaling by multiplying inv_diag itself,
    # but normalization will undo absolute scale. So best-effort: keep inv_diag ones.
    return (QA, QG, inv_diag)


  def _effective_diag(self, inv_diag: jnp.ndarray) -> jnp.ndarray:
    """Compute d_eff from stored inv_diag according to norm_mode."""
    if self._norm_mode == "tanh":
      c = jnp.asarray(self._tanh_c, dtype=inv_diag.dtype)
      return c * jnp.tanh(inv_diag / c)

    return _normalize_diag(inv_diag, self._norm_mode, self._eps)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}

    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}

      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          blk = layer_blocks[component]

          if self._use_alpha:
            QA, QG, inv_diag, alpha = blk
          else:
            QA, QG, inv_diag = blk
            alpha = None

          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          d_eff = self._effective_diag(inv_diag).astype(X2d.dtype)
          if alpha is not None:
            d_eff = (alpha.astype(X2d.dtype)) * d_eff

          out2d = self._ekfac_apply(QA, QG, d_eff, X2d)
          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)

    return product_dict


# -----------------------------
# 2) Separable diagonal ekfac
# -----------------------------
@jax.jit
def _apply_QdiagQt(Q: jnp.ndarray, diag: jnp.ndarray, X: jnp.ndarray) -> jnp.ndarray:
  """Apply (Q diag(diag) Q^T) @ X without materializing."""
  return Q @ (diag[:, None] * (Q.T @ X))


class SeparableEKFACBlock(EKFACBlock):
  """
  Separable EKFAC-style block (EKFAC-lite):

    A = QA diag(a) QA^T
    G = QG diag(g) QG^T
    P(X) = A X G

  Stored per kernel:
    - QA: (m,m) orthonormal-ish basis
    - QG: (n,n) orthonormal-ish basis
    - a: (m,) diagonal (can be indefinite)
    - g: (n,) diagonal (can be indefinite)

  Bias blocks unchanged (vector diag).

  Identity init:
    choose QA=I, QG=I, a=identity_scale * 1, g=1
    so train_matrix_product returns identity_scale * X (then base class divides by identity_scale).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0):
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape):
    if len(shape) != 2:
      raise ValueError(f"SeparableEKFACBlock.identity_block_init expects (m,n), got {shape}")
    m, n = shape
    QA = jnp.eye(m, dtype=jnp.float32)
    QG = jnp.eye(n, dtype=jnp.float32)

    # Make train_matrix_product equal to identity_scale * X:
    a = self._identity_scale * jnp.ones((m,), dtype=jnp.float32)
    g = jnp.ones((n,), dtype=jnp.float32)

    return (QA, QG, a, g)

  def train_matrix_product(self, blocks, vectors):
    product_dict = {}

    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}

      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          QA, QG, a, g = layer_blocks[component]

          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          a_eff = a.astype(X2d.dtype)  # (m,)
          g_eff = g.astype(X2d.dtype)  # (n,)

          # Left apply: A @ X
          AX = _apply_QdiagQt(QA, a_eff, X2d)

          # Right apply: X @ G (use symmetry trick: XG = (G @ X^T)^T)
          XGt = _apply_QdiagQt(QG, g_eff, AX.T)  # (n,m)
          out2d = XGt.T                          # (m,n)

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)

    return product_dict


# -----------------------------
# 3) Symmetry enforce variants
# -----------------------------
@jax.jit
def _psd_diag_from_raw(raw: jnp.ndarray, eps: float) -> jnp.ndarray:
  """
  Map unconstrained raw -> strictly positive diagonal entries.
  Using square keeps it simple and fast; eps prevents exact zeros.
  """
  return raw * raw + eps


@jax.jit
def _ekfac_apply_psd(QA: jnp.ndarray,
                     QG: jnp.ndarray,
                     raw_s: jnp.ndarray,
                     X2d: jnp.ndarray,
                     eps: float) -> jnp.ndarray:
  """
  Apply P@X where
    P = (QG ⊗ QA) diag(s^2) (QG ⊗ QA)^T   (PSD)
  without forming Kronecker matrices.

  raw_s: (m*n,) parameters; effective diag = raw_s^2 + eps
  """
  m, n = X2d.shape
  # rotate into eigenbasis
  X_hat = (QA.T @ X2d) @ QG                 # (m, n)

  # apply PSD diagonal in that basis
  diag = _psd_diag_from_raw(raw_s, eps)     # (m*n,)
  X_hat = (diag * X_hat.reshape(-1)).reshape((m, n))

  # rotate back
  return (QA @ X_hat) @ QG.T


@jax.jit
def _psd_from_raw(raw: jnp.ndarray, eps: float) -> jnp.ndarray:
  """PSD diagonal entries from unconstrained raw parameters."""
  return raw * raw + eps


class PSDEKFACBlock(EKFACBlock):
  """
  EKFAC-style block, but PSD enforced like PSDSymKroneckerBlock:

    inv_precond ≈ (Q ⊗ Q) diag(s^2) (Q ⊗ Q)^T

  where diag(s^2) is always >= eps, so the operator is symmetric PSD.

  Stored per kernel:
    - QA: (m,m) orthonormal-ish basis
    - QG: (n,n) orthonormal-ish basis
    - raw_s: (m*n,) unconstrained parameters; effective diag = raw_s^2 + eps

  Bias blocks unchanged (vector diag).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               *,
               eps: float = 1e-12):
    self._eps = float(eps)
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape):
    """
    Initialize so that matrix_product is identity at init (given BlockStructures divides by identity_scale).

    We want train_matrix_product to output identity_scale * X, so effective diag in EKFAC basis should be identity_scale.
    Here diag = raw_s^2 + eps, so choose raw_s = sqrt(identity_scale) and eps small.
    """
    if len(shape) != 2:
      raise ValueError(f"PSDEKFACBlock.identity_block_init expects (m,n), got {shape}")
    m, n = shape
    QA = jnp.eye(m, dtype=jnp.float32)
    QG = jnp.eye(n, dtype=jnp.float32)

    # raw_s^2 ≈ identity_scale => raw_s = sqrt(identity_scale)
    raw_s = jnp.sqrt(jnp.asarray(self._identity_scale, dtype=jnp.float32)) * jnp.ones((m * n,), dtype=jnp.float32)
    return (QA, QG, raw_s)


  def train_matrix_product(self, blocks, vectors):
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}

      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          QA, QG, raw_s = layer_blocks[component]

          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)
          out2d = _ekfac_apply_psd(QA, QG, raw_s, X2d, eps=self._eps)

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)

    return product_dict


class PSDSeparableEKFACBlock(EKFACBlock):
  """
  PSD-separable EKFAC-style block:

    A = QA diag(a) QA^T,   a = raw_a^2 + eps
    G = QG diag(g) QG^T,   g = raw_g^2 + eps

    P(X) = A X G

  Stored per kernel:
    - QA: (m,m)
    - QG: (n,n)
    - raw_a: (m,)  -> a = raw_a^2 + eps
    - raw_g: (n,)  -> g = raw_g^2 + eps

  Bias blocks unchanged (vector diag).

  Identity init:
    choose raw_a = sqrt(identity_scale), raw_g = 1
    so train_matrix_product returns identity_scale * X (then base class divides by identity_scale).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               *,
               eps: float = 1e-12):
    self._eps = float(eps)
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  def identity_block_init(self, shape):
    """
    Initialize (QA, QG, raw_a, raw_g) such that P acts like identity_scale * I at train_matrix_product time.

    Set QA=I, QG=I.
    Want A = identity_scale * I and G = I, so that A X G = identity_scale * X.
    That makes overall matrix_product = (identity_scale*X)/identity_scale = X (identity).
    """
    if len(shape) != 2:
      raise ValueError(f"PSDSeparableEKFACBlock.identity_block_init expects (m,n), got {shape}")
    m, n = shape
    QA = jnp.eye(m, dtype=jnp.float32)
    QG = jnp.eye(n, dtype=jnp.float32)

    raw_a = jnp.sqrt(jnp.asarray(self._identity_scale, dtype=jnp.float32)) * jnp.ones((m,), dtype=jnp.float32)
    raw_g = jnp.ones((n,), dtype=jnp.float32)  # so g ≈ 1

    return (QA, QG, raw_a, raw_g)


  def train_matrix_product(self, blocks, vectors):
    product_dict = {}

    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}

      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          QA, QG, raw_a, raw_g = layer_blocks[component]

          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          a = _psd_from_raw(raw_a, self._eps).astype(X2d.dtype)  # (m,)
          g = _psd_from_raw(raw_g, self._eps).astype(X2d.dtype)  # (n,)

          # Left apply: A @ X
          AX = _apply_QdiagQt(QA, a, X2d)

          # Right apply: X @ G (use symmetry trick: XG = (G @ X^T)^T)
          XGt = _apply_QdiagQt(QG, g, AX.T)   # (n,m)
          out2d = XGt.T                        # (m,n)

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

        else:
          raise ValueError(f"Unknown block type for {component} in layer {layer_name}")

      product_dict[layer_name] = unflatten_dict(layer_product)

    return product_dict


# -----------------------------
# 4) Householder orthogonal ekfac
# -----------------------------
@jax.jit
def _safe_unit(v: jnp.ndarray, eps: float) -> jnp.ndarray:
  """u = v / (||v|| + eps). If v==0 => u==0 => Householder becomes identity."""
  norm = jnp.linalg.norm(v)
  return v / (norm + eps)


@jax.jit
def _householder_apply_one(v: jnp.ndarray, X: jnp.ndarray, eps: float) -> jnp.ndarray:
  """
  Apply a single Householder reflection H(v) to X:
    X <- (I - 2 u u^T) X
  where u = v/(||v||+eps).
  X shape: (n, d)
  v shape: (n,)
  """
  u = _safe_unit(v, eps)  # (n,)
  # proj = u^T X : (d,)
  proj = jnp.dot(u, X)    # (d,)
  # X - 2 u proj^T
  return X - 2.0 * (u[:, None] * proj[None, :])


def _apply_householder_stack(V: jnp.ndarray, X: jnp.ndarray, eps: float, reverse: bool) -> jnp.ndarray:
  """
  Apply Q X where Q = Π_i H(V[i]).
  If reverse=True, applies reflections in reverse order, which equals Q^T since H is symmetric.
  V shape: (k, n)
  X shape: (n, d)
  """
  V_eff = V[::-1] if reverse else V

  def body(X_carry, v):
    return _householder_apply_one(v, X_carry, eps), None

  X_out, _ = jax.lax.scan(body, X, V_eff)
  return X_out


@jax.jit
def _apply_QDQ(V: jnp.ndarray, d: jnp.ndarray, X: jnp.ndarray, eps: float) -> jnp.ndarray:
  """
  Apply P X with P = Q diag(d) Q^T, Q = product of Householders from V.
  V: (k, n), d: (n,), X: (n, d2)
  """
  # Q^T X
  Xh = _apply_householder_stack(V, X, eps=eps, reverse=True)
  # diag(d) (Q^T X)
  Xh = d[:, None] * Xh
  # Q (diag(d) Q^T X)
  return _apply_householder_stack(V, Xh, eps=eps, reverse=False)


class HouseholderDiagKroneckerBlock(KroneckerBlock):
  """
  EKFAC-like *basis learning* but cheaper than full eigenbases:

    P = Q diag(d) Q^T,  Q = Π_{i=1..k} Householder(v_i)

  Storage per factor (size n):
    - V: (k, n)  Householder vectors
    - d: (n,)    diagonal in that basis

  Symmetry:
    - enforced by construction. No need to store upper/lower separately.

  Notes:
    - PSD if d >= 0 (you can enforce via update rule / projection if desired).
    - Indefinite allowed if d has mixed signs (your “inverting direction can help” regime).
  """

  def __init__(self,
               network_params,
               layer_names,
               block_structure_init,
               rank=None,
               identity_scale=1.0,
               *,
               num_reflections: int = 4,  #keep small for compute efficiency
               eps: float = 1e-8):
    self._num_reflections = int(num_reflections)
    self._eps = float(eps)
    super().__init__(network_params, layer_names, block_structure_init, rank, identity_scale)

  # ---- factor init ----
  def identity_block_init(self, dim: int):
    """
    Initialize (V, d) so that P = identity_scale * I.

    - V = 0 => each Householder is identity (safe normalization => u=0)
    - d = identity_scale * 1  => P = identity_scale I
    """
    V = jnp.zeros((self._num_reflections, dim), dtype=jnp.float32)
    d = (self._identity_scale * jnp.ones((dim,), dtype=jnp.float32))
    return V, d

  def _make_blocks(self, network_params, layer_names, initialization=False):
    """
    Override because each kernel factor is (V, d), not a single matrix.
    """
    blocks = {}
    for layer_name in layer_names:
      flat_params = flatten_dict(network_params[layer_name])
      layer_blocks = {}

      for key in flat_params.keys():
        if any(k in key for k in KERNEL_KEYS):
          kernel = flat_params[key]
          if not hasattr(kernel, "shape"):
            raise ValueError(f"Kernel for layer {layer_name} does not have a shape attribute")

          if len(kernel.shape) == 2:
            m, n = kernel.shape
          elif len(kernel.shape) == 4:
            k_h, k_w, cin, cout = kernel.shape
            m, n = k_h * k_w * cin, cout
          else:
            raise ValueError(f"Kernel shape {kernel.shape} not supported for layer {layer_name}")

          # Left factor acts on rows (m), right factor on cols (n)
          FA = self.identity_block_init(m)  # (V_A, d_A)
          FB = self.identity_block_init(n)  # (V_B, d_B)
          layer_blocks[key] = (FA, FB)

        if any(k in key for k in BIAS_KEYS):
          bias = flat_params[key]
          if not hasattr(bias, "shape"):
            raise ValueError(f"Bias for layer {layer_name} does not have a shape attribute")
          flat_bias, _ = ravel_pytree(bias)
          layer_blocks[key] = self.identity_diag_init(flat_bias.shape[0])

      blocks[layer_name] = layer_blocks
    return blocks

  # ---- apply ----
  def train_matrix_product(self, blocks, vectors):
    """
    Kernel: apply vec( A X B^T ) where:
      A = Q_A diag(d_A) Q_A^T
      B = Q_B diag(d_B) Q_B^T

    Implemented as:
      AX = A @ X
      out = (B @ AX^T)^T  (since B symmetric)
    """
    product_dict = {}
    for layer_name, layer_vectors in vectors.items():
      layer_blocks = blocks[layer_name]
      layer_product = {}

      for component, vec_component in flatten_dict(layer_vectors).items():
        if any(k in component for k in KERNEL_KEYS):
          (V_A, d_A), (V_B, d_B) = layer_blocks[component]

          X2d, orig_shape = _reshape_kernel_to_2d(vec_component)

          # Left apply: (m,m) operator on rows
          AX = _apply_QDQ(V_A, d_A.astype(X2d.dtype), X2d, eps=self._eps)

          # Right apply using transpose trick: X @ B  == (B @ X^T)^T
          out2d = _apply_QDQ(V_B, d_B.astype(X2d.dtype), AX.T, eps=self._eps).T

          layer_product[component] = out2d.reshape(orig_shape)

        elif any(k in component for k in BIAS_KEYS):
          diag = layer_blocks[component]
          layer_product[component] = diag * vec_component

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