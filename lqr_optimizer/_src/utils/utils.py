""" Various utilities functions for LQR optimization"""
import time
import os
import pickle
import signal
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from aim import Distribution

import flax.linen as nn
from flax.linen import Sequential
from flax.core.frozen_dict import FrozenDict
from flax import struct
from flax.training import train_state
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import List, Tuple, Any, Dict, Optional, TypedDict, Callable, NamedTuple
from types import FrameType
from pathlib import Path
from collections.abc import Sequence

from jax.tree_util import Partial

from lqr_optimizer._src.utils.grokking_dataset import ModSumDataset, ModDivisionDataset, ModSubtractDataset, ModMulDataset, ModExpDataset, PermutationGroup, load_grok_ds
from lqr_optimizer._src.utils.precond_optimizers import nonlinear_cg

def vjp_f(f, x):
  """ Return the vjp in a form that can be applied directly over a vector
  """
  _, f = jax.vjp(f, x)
  return lambda v: f(v)[0]


def add_f(f, g):
  """ Return the function composition of f added to g
  """
  return lambda x: f(x) + g(x)

def subtract_f(f, g):
  """ Return the function composition of g subtracted to f
  """
  return lambda x: f(x) - g(x)

def normalize_gradient(gradient):
  # Compute the total L2 norm of the gradient using jnp.linalg.norm
  total_norm = jnp.linalg.norm(ravel_pytree(gradient)[0], ord=2)

  # Avoid division by zero
  total_norm = jnp.maximum(total_norm, 1e-9)

  # Normalize each gradient component
  normalized_gradient = jax.tree_util.tree_map(lambda g: g / total_norm, gradient)

  return normalized_gradient

def clip_gradient(gradient, clip_norm):
  example_norm = jnp.linalg.norm(gradient)
  clipped_grad = gradient * (clip_norm / jnp.maximum(example_norm, clip_norm))
  return clipped_grad

def clip_norm_single_example(_grad, clip_norm):
  """Apply clipping norm to a single example within a batch"""
  ravel_grad, unravel_fn = ravel_pytree(_grad)
  clipped_grad = clip_gradient(ravel_grad, clip_norm)
  return unravel_fn(clipped_grad)

vmapped_clip_norm = jax.vmap(clip_norm_single_example, in_axes=(0, None))
def treemapped_clip_norm(gradient, clip_norm):
  return jax.tree_map(Partial(clip_gradient, clip_norm=clip_norm), gradient)

def treemapped_clip_element_wise(gradient, clip_value):
  return jax.tree_map(Partial(jnp.clip, min=-1*clip_value, max=clip_value), gradient)


@jax.jit
def pytree_max_min(pytree):
  leaves = jax.tree_util.tree_leaves(pytree)  # Extract leaves
  leaves = [jnp.ravel(leaf) for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.number)]  # Flatten each array
  return (jnp.max(jnp.concatenate(leaves)), jnp.min(jnp.concatenate(leaves))) if leaves else (None, None)  # Get max if non-empty


@jax.jit
def pytree_l2_norm(pytree):
  leaves = jax.tree_util.tree_leaves(pytree)  # Extract leaves
  leaves = [jnp.ravel(leaf) for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.number)]  # Flatten each array
  return jnp.linalg.norm(jnp.concatenate(leaves)) if leaves else jnp.array(0.0)  # Compute L2 norm


@jax.jit
def ravel_pytree_l2_norm(pytree):
  vector, _ = ravel_pytree(pytree)
  return jnp.linalg.norm(vector)


@jax.jit
def get_per_layer_norm(precond):
  return {_layer: ravel_pytree_l2_norm(_block) for _layer, _block in precond.items()}

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

class StageDescriptor(NamedTuple):
  name: str
  kind: str
  param_name: Optional[str] = None
  fast_path_kind: Optional[str] = None
  passive_state_hessian: Optional[str] = None


class LqrSegmentDescriptor(NamedTuple):
  name: str
  execution_stage_names: Tuple[str, ...]
  sample_separable_second_order: Optional[bool] = None


class ResolvedLqrSegmentDescriptor(NamedTuple):
  name: str
  start_index: int
  stop_index: int
  execution_stage_descriptors: Tuple[StageDescriptor, ...]
  controlled_param_names: Tuple[str, ...]
  sample_separable_second_order: Optional[bool] = None


_ALLOWED_STAGE_KINDS = frozenset(("controlled", "passive"))
_ALLOWED_FAST_PATH_KINDS = frozenset((None, "linear_controlled", "piecewise_linear_passive"))
_ALLOWED_PASSIVE_STATE_HESSIANS = frozenset((None, "zero", "generic"))


def make_controlled_stage_descriptor(name: str, param_name: str, fast_path_kind: Optional[str] = None) -> StageDescriptor:
  return StageDescriptor(name=name, kind="controlled", param_name=param_name, fast_path_kind=fast_path_kind)


def make_passive_stage_descriptor(name: str, fast_path_kind: Optional[str] = None,
                                  passive_state_hessian: Optional[str] = None) -> StageDescriptor:
  if passive_state_hessian is None and fast_path_kind == "piecewise_linear_passive":
    passive_state_hessian = "zero"
  return StageDescriptor(
    name=name,
    kind="passive",
    param_name=None,
    fast_path_kind=fast_path_kind,
    passive_state_hessian=passive_state_hessian,
  )


def make_lqr_segment_descriptor(name: str, execution_stage_names: Tuple[str, ...],
                                sample_separable_second_order: Optional[bool] = None) -> LqrSegmentDescriptor:
  return LqrSegmentDescriptor(
    name=name,
    execution_stage_names=tuple(execution_stage_names),
    sample_separable_second_order=sample_separable_second_order,
  )


def validate_stage_descriptors(stage_descriptors: Tuple[StageDescriptor, ...], *, num_layers: int) -> Tuple[StageDescriptor, ...]:
  if len(stage_descriptors) != num_layers:
    raise ValueError(
      f"Stage descriptor count {len(stage_descriptors)} does not match layer count {num_layers}."
    )

  stage_names = set()
  controlled_param_names = set()
  for descriptor in stage_descriptors:
    if descriptor.kind not in _ALLOWED_STAGE_KINDS:
      raise ValueError(f"Unknown stage kind '{descriptor.kind}' for stage '{descriptor.name}'.")
    if descriptor.name in stage_names:
      raise ValueError(f"Duplicate execution stage name '{descriptor.name}'.")
    stage_names.add(descriptor.name)

    if descriptor.fast_path_kind not in _ALLOWED_FAST_PATH_KINDS:
      raise ValueError(
        f"Unknown fast_path_kind '{descriptor.fast_path_kind}' for stage '{descriptor.name}'."
      )
    if descriptor.passive_state_hessian not in _ALLOWED_PASSIVE_STATE_HESSIANS:
      raise ValueError(
        f"Unknown passive_state_hessian '{descriptor.passive_state_hessian}' for stage '{descriptor.name}'."
      )

    if descriptor.kind == "controlled":
      if descriptor.param_name is None:
        raise ValueError(f"Controlled stage '{descriptor.name}' must define param_name.")
      if descriptor.param_name in controlled_param_names:
        raise ValueError(f"Duplicate controlled param_name '{descriptor.param_name}'.")
      controlled_param_names.add(descriptor.param_name)
      if descriptor.fast_path_kind == "piecewise_linear_passive":
        raise ValueError(
          f"Controlled stage '{descriptor.name}' cannot use passive fast_path_kind '{descriptor.fast_path_kind}'."
        )
      if descriptor.passive_state_hessian is not None:
        raise ValueError(
          f"Controlled stage '{descriptor.name}' cannot define passive_state_hessian."
        )
    else:
      if descriptor.param_name is not None:
        raise ValueError(f"Passive stage '{descriptor.name}' must not define param_name.")
      if descriptor.fast_path_kind == "linear_controlled":
        raise ValueError(
          f"Passive stage '{descriptor.name}' cannot use controlled fast_path_kind '{descriptor.fast_path_kind}'."
        )
      if descriptor.fast_path_kind == "piecewise_linear_passive" and descriptor.passive_state_hessian != "zero":
        raise ValueError(
          f"Passive stage '{descriptor.name}' with piecewise_linear_passive fast_path_kind must declare zero state Hessian."
        )

  return stage_descriptors


def validate_lqr_segment_descriptors(lqr_segment_descriptors: Tuple[LqrSegmentDescriptor, ...],
                                     execution_stage_descriptors: Tuple[StageDescriptor, ...]
                                     ) -> Tuple[ResolvedLqrSegmentDescriptor, ...]:
  stage_index_by_name = {stage.name: index for index, stage in enumerate(execution_stage_descriptors)}
  segment_names = set()
  covered_stage_names = set()
  expected_start = 0
  resolved_segments = []

  if not lqr_segment_descriptors and execution_stage_descriptors:
    raise ValueError("LLQR segment descriptors must not be empty when execution stages exist.")

  for segment in lqr_segment_descriptors:
    if not segment.name:
      raise ValueError("LLQR segment names must be non-empty.")
    if segment.name in segment_names:
      raise ValueError(f"Duplicate LLQR segment name '{segment.name}'.")
    if (
        segment.sample_separable_second_order is not None
        and not isinstance(segment.sample_separable_second_order, bool)
    ):
      raise ValueError(
        f"LLQR segment '{segment.name}' has invalid sample_separable_second_order policy "
        f"'{segment.sample_separable_second_order}'. Expected None, True, or False."
      )
    segment_names.add(segment.name)

    execution_stage_names = tuple(segment.execution_stage_names)
    if not execution_stage_names:
      raise ValueError(f"LLQR segment '{segment.name}' must reference at least one execution stage.")

    segment_indices = []
    for stage_name in execution_stage_names:
      if stage_name not in stage_index_by_name:
        raise ValueError(
          f"LLQR segment '{segment.name}' references unknown execution stage '{stage_name}'."
        )
      if stage_name in covered_stage_names:
        raise ValueError(
          f"Execution stage '{stage_name}' appears in more than one LLQR segment."
        )
      covered_stage_names.add(stage_name)
      segment_indices.append(stage_index_by_name[stage_name])

    expected_indices = list(range(expected_start, expected_start + len(segment_indices)))
    if segment_indices != expected_indices:
      raise ValueError(
        f"LLQR segment '{segment.name}' must cover a contiguous execution-stage slice in forward order."
      )

    segment_stage_descriptors = tuple(execution_stage_descriptors[index] for index in segment_indices)
    controlled_param_names = tuple(
      stage.param_name for stage in segment_stage_descriptors if stage.kind == "controlled"
    )
    resolved_segments.append(
      ResolvedLqrSegmentDescriptor(
        name=segment.name,
        start_index=expected_start,
        stop_index=expected_start + len(segment_indices),
        execution_stage_descriptors=segment_stage_descriptors,
        controlled_param_names=controlled_param_names,
        sample_separable_second_order=segment.sample_separable_second_order,
      )
    )
    expected_start += len(segment_indices)

  if expected_start != len(execution_stage_descriptors):
    raise ValueError("LLQR segments must cover every execution stage exactly once.")

  return tuple(resolved_segments)


class EnhancedSequential(nn.Module):
  layers: List[nn.Module]
  stage_descriptors: Optional[Tuple[StageDescriptor, ...]] = None
  lqr_segment_descriptors: Optional[Tuple[LqrSegmentDescriptor, ...]] = None
  legacy_checkpoint_migrator: Optional[Callable[[Any, Any, Any, Any], Tuple[Any, Any]]] = None

  def __call__(self, x: Any) -> Any:
    """Applies the blocks sequentially to the input."""
    for block in self.layers:
      x = block(x)
    return x

  @property
  def execution_stage_descriptors(self) -> Tuple[StageDescriptor, ...]:
    if self.stage_descriptors is None:
      stage_descriptors = tuple(
        make_controlled_stage_descriptor(name=f"layers_{i}", param_name=f"layers_{i}", fast_path_kind=None)
        for i, _ in enumerate(self.layers)
      )
    else:
      stage_descriptors = tuple(self.stage_descriptors)
    return validate_stage_descriptors(stage_descriptors, num_layers=len(self.layers))

  def validate_stage_descriptors(self) -> Tuple[StageDescriptor, ...]:
    stage_descriptors = self.execution_stage_descriptors
    self.validate_lqr_segment_descriptors()
    return stage_descriptors

  @property
  def resolved_lqr_segment_descriptors(self) -> Tuple[ResolvedLqrSegmentDescriptor, ...]:
    execution_stage_descriptors = self.execution_stage_descriptors
    if self.lqr_segment_descriptors is None:
      if self.stage_descriptors is not None:
        raise ValueError(
          "Models with explicit execution stage descriptors must also provide explicit LLQR segment descriptors."
        )
      lqr_segment_descriptors = tuple(
        make_lqr_segment_descriptor(stage.name, (stage.name,))
        for stage in execution_stage_descriptors
      )
    else:
      lqr_segment_descriptors = tuple(self.lqr_segment_descriptors)
    return validate_lqr_segment_descriptors(lqr_segment_descriptors, execution_stage_descriptors)

  def validate_lqr_segment_descriptors(self) -> Tuple[ResolvedLqrSegmentDescriptor, ...]:
    return self.resolved_lqr_segment_descriptors

  @property
  def controlled_stage_descriptors(self) -> Tuple[StageDescriptor, ...]:
    return tuple(stage for stage in self.execution_stage_descriptors if stage.kind == "controlled")

  @property
  def controlled_stage_names(self) -> Tuple[str, ...]:
    return tuple(stage.param_name for stage in self.controlled_stage_descriptors)

  @property
  def has_passive_stages(self) -> bool:
    return any(stage.kind == "passive" for stage in self.execution_stage_descriptors)

  def get_execution_stage_index(self, stage_name: str) -> int:
    for index, stage in enumerate(self.execution_stage_descriptors):
      if stage.name == stage_name:
        return index
    raise ValueError(f"Execution stage '{stage_name}' not found.")

  def get_controlled_stage_execution_index(self, param_name: str) -> int:
    for index, stage in enumerate(self.execution_stage_descriptors):
      if stage.param_name == param_name:
        return index
    raise ValueError(f"Controlled stage '{param_name}' not found.")

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
    """Applies a controlled stage using its explicit stage-to-execution mapping."""
    block = self.layers[self.get_controlled_stage_execution_index(block_name)]
    block_params = params.get(block_name, {})
    return block.apply({"params": block_params}, x)

  def apply_execution_stage_from_name(self, stage_name: str, x: Any, variables: FrozenDict) -> Any:
    """Applies one execution stage identified by its explicit stage name."""
    block = self.layers[self.get_execution_stage_index(stage_name)]
    return block.apply(variables, x)

  def apply_block_from_params(self, block_params: FrozenDict, x: Any, index) -> Any:
    block = self.layers[index]
    return block.apply(block_params, x)

  def maybe_migrate_legacy_checkpoint(self, loaded_params, loaded_batch_stats,
                                      init_params, init_batch_stats):
    # Checkpoint round-trips can reorder top-level FrozenDict keys while preserving
    # the actual split-stage schema. Treat key-set equality as a same-layout match.
    params_match = set(loaded_params.keys()) == set(init_params.keys())
    batch_stats_match = set(loaded_batch_stats.keys()) == set(init_batch_stats.keys())
    if params_match and batch_stats_match:
      return loaded_params, loaded_batch_stats, False
    if self.legacy_checkpoint_migrator is None:
      return loaded_params, loaded_batch_stats, False
    migrated_params, migrated_batch_stats = self.legacy_checkpoint_migrator(
      loaded_params, loaded_batch_stats, init_params, init_batch_stats
    )
    return migrated_params, migrated_batch_stats, True

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


##################################
# Loading utils
##################################
def load_main_optimizer(cfg, lr_or_sched):
  if cfg.main_optimizer == "polyak":
    model_optimizer = optax.sgd(learning_rate=lr_or_sched, momentum=cfg.momentum)
  elif cfg.main_optimizer == "adam":
    model_optimizer = optax.adam(learning_rate=lr_or_sched)
  elif cfg.main_optimizer == "sgd":
    model_optimizer = optax.sgd(learning_rate=lr_or_sched)
  elif cfg.main_optimizer == "adamw_b2-98": # Grokking exps #TODO: move to config files...
    model_optimizer = optax.adamw(learning_rate=lr_or_sched, b2=0.98, weight_decay=cfg.weight_decay)
  elif cfg.main_optimizer == "adamw_b2-95": # GPT experiments
    model_optimizer = optax.adamw(learning_rate=lr_or_sched, b2=0.95, weight_decay=cfg.weight_decay)
  else:
    raise ValueError("Unknown main optimizer")
  return model_optimizer


def load_precond_optimizer(cfg, lr):
  optax_solver_for_precond = []
  if cfg.precond_clip_norm:
    optax_solver_for_precond.append(clip_by_group_norm(cfg.precond_clip_norm))
  if cfg.precond_clip_element_wise:
    optax_solver_for_precond.append(optax.clip(cfg.precond_clip_element_wise))
  if cfg.precond_solver == "adam":
    optax_solver_for_precond.append(optax.adam(lr))
  elif cfg.precond_solver == "momentum":
    optax_solver_for_precond.append(optax.sgd(lr, momentum=cfg.momentum))
  elif cfg.precond_solver == "sgd":
    optax_solver_for_precond.append(optax.sgd(lr))
  elif cfg.precond_solver == "cg_zoom_hz":
    opt = nonlinear_cg(
      linesearch="optax_zoom",
      method="hz",
      optax_ls_kwargs=dict(max_linesearch_steps=20),
    )
    optax_solver_for_precond.append(opt)
  elif cfg.precond_solver == "cg_back_pr+":
    opt = nonlinear_cg(
      linesearch="optax_backtracking",
      method="pr+",
      # enforce_descent=False,
      optax_ls_kwargs=dict(max_backtracking_steps=20),
    )
    optax_solver_for_precond.append(opt)
  else:
    raise ValueError("Unknown precond optimizer")
  return optax.chain(*optax_solver_for_precond)


def precond_solver_requires_value_and_grad(precond_solver_name):
  """Return whether the preconditioner optimizer needs objective values."""
  if precond_solver_name in ("adam", "momentum", "sgd"):
    return False
  if precond_solver_name in ("cg_zoom_hz", "cg_back_pr+"):
    return True
  raise ValueError("Unknown precond optimizer")


def clip_by_group_norm(max_norm: float) -> optax.GradientTransformation:
  """
  Clip each gradient leaf independently by its L2 norm.

  Args:
      max_norm: Maximum allowed norm for each parameter leaf.

  Returns:
      An optax.GradientTransformation that can be used in optax.chain.
  """

  def init_fn(params):
    # No state needed for clipping
    return ()

  def update_fn(updates, state, params=None):
    def clip_leaf(g: jnp.ndarray) -> jnp.ndarray:
      leaf_norm = optax._src.linear_algebra.global_norm(g) # Safer than jnp.linalg.norm
      leaf_norm = jnp.maximum(max_norm, leaf_norm)
      scale = max_norm / leaf_norm
      return g * scale

    # Treemap over all leaves
    clipped_updates = jax.tree_map(clip_leaf, updates)
    return clipped_updates, state

  return optax.GradientTransformation(init_fn, update_fn)

##################################
# Training utils
##################################
# Simple cross-entropy loss for classification
def cross_entropy_loss(log_probs, y, label_smoothing=0.0):
  """
  Cross-entropy loss with optional label smoothing.

  Args:
      log_probs: [batch, num_classes], log probabilities (e.g., from nn.log_softmax)
      y: [batch,] integer class labels
      label_smoothing: float in [0, 1]. 0 = no smoothing (standard CE)
  """
  num_classes = log_probs.shape[-1]
  one_hot = jax.nn.one_hot(y, num_classes)

  if label_smoothing > 0.0:
    smooth = label_smoothing / num_classes
    one_hot = (1.0 - label_smoothing) * one_hot + smooth

  # Negative log-likelihood
  nll = -jnp.sum(one_hot * log_probs, axis=-1)
  return jnp.mean(nll)


def loss_eval(state, x, y):
  """
  params: parameters of the model
  apply_fn: function to apply model (Flax typically provides model.apply)
  x: input data
  y: integer class labels
  """
  log_probs = state.apply_inf_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    x,
    mutable=False
  )  # shape [batch_size, num_classes]
  # We expect the final layer to produce log-softmax output (as defined in MLP),

  return cross_entropy_loss(log_probs, y)


def apply_cutout(image, size=16, p=0.5):
  """
  Randomly masks out a square patch in the image with probability `p`.
  Works with TensorFlow tensors of shape [H, W, C].
  """
  def cutout_fn():
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    center_x = tf.random.uniform([], 0, w, dtype=tf.int32)
    center_y = tf.random.uniform([], 0, h, dtype=tf.int32)

    half_size = size // 2
    x1 = tf.clip_by_value(center_x - half_size, 0, w)
    y1 = tf.clip_by_value(center_y - half_size, 0, h)
    x2 = tf.clip_by_value(center_x + half_size, 0, w)
    y2 = tf.clip_by_value(center_y + half_size, 0, h)

    mask = tf.ones_like(image, dtype=image.dtype)
    mask = tf.tensor_scatter_nd_update(mask,
        indices=tf.reshape(tf.stack(tf.meshgrid(tf.range(y1, y2), tf.range(x1, x2), indexing='ij'), -1), [-1, 2]),
        updates=tf.zeros([(y2 - y1) * (x2 - x1), tf.shape(image)[-1]], dtype=image.dtype)
    )
    return image * mask

  return tf.cond(tf.random.uniform([]) < p, cutout_fn, lambda: image)


### ImageNet-specific dataloader helper fn ####
@tf.function
def resize_imgnet_dataset(images, labels, is_training=False):
    if is_training:
        # Generate a random number to decide the resizing dimensions
        random_num = tf.random.uniform(shape=[], minval=0, maxval=1)

        if random_num < 0.5:
            # Resize images to 480x480 with 50% probability when training
            images = tf.image.resize(images, [480, 480])
        else:
            # Resize images to 256x256 with 50% probability when training
            images = tf.image.resize(images, [256, 256])
    else:
        # Resize images to 256x256 when not training
        images = tf.image.resize(images, [256, 256])
    return images, labels


@tf.function
def augment_train_imagenet_dataset_res50(image, label):
    # Randomly crop the image
    image = tf.image.random_crop(image, [224, 224, 3])  # Assuming image has 3 color channels
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)

    return image, label


@tf.function
def process_test_imagenet_dataset(image, label):
    image = tf.image.central_crop(image, 224/256)  # assuming input image size is 256x256
    return image, label


def prepare_dataloader(
    batch_size=128,
    train=True,
    dataset='mnist',
    augment_dataset=False,
    shuffle: bool = True,
    lt_config=None,  # e.g., {"imbalance_ratio": 100, "distribution": "exp", "seed": 0}
    dataset_dir: str = None,
    batch_overlap_fraction: float = 0.0,   # NEW (0.0 means disabled)
):
  """
  Creates a generator that yields (x, y) from the specified dataset:
    - MNIST, truncated_mnist, CIFAR-10, CIFAR-100
    - If lt_config is provided and dataset is CIFAR-10/100 (train=True),
      builds a long-tailed (LT) training split (CIFAR-10-LT / CIFAR-100-LT).

  lt_config (dict or None):
    - "imbalance_ratio": float, optional. E.g., 100 means max:min = 100:1.
      (If omitted, defaults to 100.)
    - OR "imb_factor": float in (0,1], optional. (imb_factor = 1/imbalance_ratio)
    - "distribution": str, optional. Only "exp" is implemented (default).
    - "seed": int, optional. Controls per-class shuffles before taking.
  Returns:
    (generator, info)
      info["num_classes"], info["ds_size"],
      and if LT is used: info["class_counts"] (list length = num_classes).
  """
  ########
  # Overlap config
  overlap_frac = float(batch_overlap_fraction or 0.0)
  if not (0.0 <= overlap_frac < 1.0):
    raise ValueError(f"batch_overlap_fraction must be in [0,1), got {overlap_frac}")

  # replace_size = how many new examples per step (stride)
  replace = int(round(batch_size * (1.0 - overlap_frac)))

  # ensure valid
  replace = max(1, min(batch_size, replace))
  enable_overlap = train and (replace < batch_size)  # only meaningful for train
  ########
  grokking_datasets = {
    'mod_sum': lambda: ModSumDataset(frac_train=0.6, p=97, k=5),
    'mod_subtract': lambda: ModSubtractDataset(frac_train=0.6, p=97, k=5),
    'mod_mul': lambda: ModMulDataset(frac_train=0.6, p=97, k=5),
    'mod_division': lambda: ModDivisionDataset(frac_train=0.6, p=97, k=5),
    'mod_exp': lambda: ModExpDataset(frac_train=0.6, p=97, k=5),
    'permutation': lambda: PermutationGroup(frac_train=0.6, p=97, k=5),
  }
  if dataset in grokking_datasets:
    split = 'train' if train else 'test'
    ds = grokking_datasets[dataset]()
    iterator, info = load_grok_ds(ds, split=split, batch_size=batch_size, with_info=True)

    def generator():
      for x_batch, y_batch in iterator:
        yield x_batch, y_batch
    return generator(), info

  # ---------- Standard image datasets ----------
  info = {}
  if dataset in ('mnist', 'truncated_mnist'):
    ds_name = 'mnist'
    mean = 0.1307
    std = 0.3081
    info["num_classes"] = 10
  elif dataset == 'cifar-10':
    ds_name = 'cifar10'
    mean = jnp.array([0.4914, 0.4822, 0.4465])
    std = jnp.array([0.2470, 0.2435, 0.2616])
    info["num_classes"] = 10
  elif dataset == 'cifar-100':
    ds_name = 'cifar100'
    mean = jnp.array([0.5071, 0.4867, 0.4408])
    std = jnp.array([0.2675, 0.2565, 0.2761])
    info["num_classes"] = 100
  elif dataset == 'imagenet':
    # ImageNet-1k normalization
    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
    info["num_classes"] = 1000

    # Build from TFDS builder to select 'train' vs 'validation' correctly.
    builder = tfds.builder("imagenet2012")
    # Assumes data present locally already; this is idempotent if already prepared.
    builder.download_and_prepare(download_dir=dataset_dir)
    split = "train" if train else "validation"
    split = tfds.split_for_jax_process(split, drop_remainder=True)

    # Load dataset (supervised: (image, label))
    ds = builder.as_dataset(
      split=split,
      as_supervised=True,
      shuffle_files=True,
      read_config=tfds.ReadConfig(try_autocache=False, skip_prefetch=True),
    )

    # Dataset size
    info["ds_size"] = int(builder.info.splits[split].num_examples)

    # Cache & shuffle like the other branches
    # ds = ds.cache() Oh, no, not caching imagenet...
    if train and shuffle:
      ds = ds.shuffle(4096, seed=0, reshuffle_each_iteration=True)

    # Pre-processing & augmentation
    # We apply resize+crop BEFORE batching (per-example ops).
    def _resize_train(x, y):
      return resize_imgnet_dataset(x, y, is_training=True)

    def _resize_eval(x, y):
      return resize_imgnet_dataset(x, y, is_training=False)

    if train and augment_dataset:
      ds = ds.map(_resize_train, num_parallel_calls=tf.data.AUTOTUNE)
      ds = ds.map(augment_train_imagenet_dataset_res50, num_parallel_calls=tf.data.AUTOTUNE)
    else:
      ds = ds.map(_resize_eval, num_parallel_calls=tf.data.AUTOTUNE)
      ds = ds.map(process_test_imagenet_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch → prefetch → repeat
    # Batch (overlapped or standard)
    if enable_overlap:
      if batch_size % replace != 0:
        raise ValueError(
          f"With overlap enabled, require batch_size % replace == 0. "
          f"Got batch_size={batch_size}, replace={replace}."
        )
      num_chunks = batch_size // replace

      # First, form microbatches of size `replace`
      ds = ds.batch(replace, drop_remainder=True)

      # Window over consecutive microbatches (overlap achieved by shift=1 microbatch)
      ds = ds.window(num_chunks, shift=1, drop_remainder=True)

      # IMPORTANT: window returns (x_window_ds, y_window_ds) => flat_map gets 2 args
      def _pack_window(xw, yw):
        # Collect `num_chunks` microbatches into tensors:
        # x: [num_chunks, replace, ...], y: [num_chunks, replace]
        xy = tf.data.Dataset.zip((xw, yw)).batch(num_chunks, drop_remainder=True)

        # Now reshape to full batch [batch_size, ...]
        def _merge(x, y):
          x = tf.reshape(x, (batch_size,) + tuple(x.shape[2:]))
          y = tf.reshape(y, (batch_size,))
          return x, y

        return xy.map(_merge, num_parallel_calls=tf.data.AUTOTUNE)

      ds = ds.flat_map(_pack_window)
    else:
      ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()

    # ---------- Generator ----------
    def generator():
      for x_batch, y_batch in ds:
        x_batch = x_batch.numpy().astype(jnp.float32) / 255.0
        x_batch = (x_batch - mean[None, None, None, :]) / std[None, None, None, :]
        yield jnp.array(x_batch, dtype=jnp.float32), jnp.array(y_batch.numpy(), dtype=jnp.int32)

    return generator(), info
    # ==============================================================================================
  else:
    raise ValueError("Unsupported dataset. Use 'mnist', 'truncated_mnist', 'cifar-10', or 'cifar-100'.")

  # Load base split
  ds, _info = tfds.load(ds_name, split='train' if train else 'test', as_supervised=True, with_info=True)

  # Optionally truncate MNIST train
  if dataset == 'truncated_mnist' and train:
    ds = ds.shuffle(buffer_size=_info.splits['train'].num_examples, seed=0)
    ds = ds.take(10_000)

  # ---------- Build LT split for CIFAR train if requested ----------
  def _make_lt_counts(num_classes, per_class_max, lt_cfg):
    # Accept either imb_factor (0<imb<=1) or imbalance_ratio (>=1)
    if lt_cfg is None:
      return [per_class_max] * num_classes
    imb_factor = lt_cfg.get("imb_factor", None)
    imb_ratio = lt_cfg.get("imbalance_ratio", None)
    if imb_factor is None:
      imb_factor = 1.0 / float(imb_ratio if imb_ratio is not None else 100.0)
    dist = lt_cfg.get("distribution", "exp")
    # Exponential long-tail (class 0 = head, class K-1 = tail)
    if dist != "exp":
      raise ValueError(f"Only 'exp' distribution is supported, got '{dist}'.")
    if num_classes == 1:
      return [per_class_max]
    counts = []
    for i in range(num_classes):
      # decay exponent spans [0,1]
      frac = i / (num_classes - 1)
      ci = int(round(per_class_max * (imb_factor ** frac)))
      counts.append(max(1, ci))
    return counts

  if train and lt_config is not None and dataset in ('cifar-10', 'cifar-100'):
    num_classes = info["num_classes"]
    # CIFAR train sizes: 5000/class for CIFAR-10, 500/class for CIFAR-100
    total_train = int(_info.splits['train'].num_examples)
    per_class_max = total_train // num_classes
    class_counts = _make_lt_counts(num_classes, per_class_max, lt_config)
    info["class_counts"] = class_counts
    info["ds_size"] = int(sum(class_counts))
    lt_seed = int(lt_config.get("seed", 0))

    # Build per-class subsets: filter -> shuffle (per class) -> take desired count
    # Note: this keeps labels intact and reshuffling happens later as well.
    per_class_ds = []
    for c in range(num_classes):
      ds_c = ds.filter(lambda x, y, c=c: tf.equal(y, c))
      # shuffle within class so you don't always take the same prefix
      ds_c = ds_c.shuffle(buffer_size=10_000, seed=lt_seed + c, reshuffle_each_iteration=False)
      ds_c = ds_c.take(class_counts[c])
      per_class_ds.append(ds_c)

    # Concatenate all per-class datasets and shuffle globally
    ds = per_class_ds[0]
    for dsc in per_class_ds[1:]:
      ds = ds.concatenate(dsc)

    # Cache & shuffle the final LT dataset
    ds = ds.cache()
    if shuffle:
      ds = ds.shuffle(info["ds_size"], seed=lt_seed, reshuffle_each_iteration=True)
  else:
    # Balanced/default path
    size = int(ds.cardinality()) if tf.data.experimental.cardinality(ds) != tf.data.experimental.UNKNOWN_CARDINALITY else int(_info.splits['train' if train else 'test'].num_examples)
    info["ds_size"] = size
    ds = ds.cache()
    if train and (shuffle):
      ds = ds.shuffle(info["ds_size"], seed=0, reshuffle_each_iteration=True)

  # Batch (overlapped or standard)
  if enable_overlap:
    if batch_size % replace != 0:
      raise ValueError(
        f"With overlap enabled, require batch_size % replace == 0. "
        f"Got batch_size={batch_size}, replace={replace}."
      )
    num_chunks = batch_size // replace

    # First, form microbatches of size `replace`
    ds = ds.batch(replace, drop_remainder=True)

    # Window over consecutive microbatches (overlap achieved by shift=1 microbatch)
    ds = ds.window(num_chunks, shift=1, drop_remainder=True)

    # IMPORTANT: window returns (x_window_ds, y_window_ds) => flat_map gets 2 args
    def _pack_window(xw, yw):
      # Collect `num_chunks` microbatches into tensors:
      # x: [num_chunks, replace, ...], y: [num_chunks, replace]
      xy = tf.data.Dataset.zip((xw, yw)).batch(num_chunks, drop_remainder=True)

      # Now reshape to full batch [batch_size, ...]
      def _merge(x, y):
        x = tf.reshape(x, (batch_size,) + tuple(x.shape[2:]))
        y = tf.reshape(y, (batch_size,))
        return x, y

      return xy.map(_merge, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.flat_map(_pack_window)
  else:
    ds = ds.batch(batch_size)

  if augment_dataset and train:
    if dataset in ('cifar-10', 'cifar-100'):
      ReflectionPadding2D = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], 'REFLECT'))
      def augment_with_cutout(x, y):
        x = ReflectionPadding2D(x)
        x = tf.image.random_crop(x, size=[tf.shape(x)[0], 32, 32, 3])
        x = tf.image.random_flip_left_right(x)
        x = tf.map_fn(lambda img: apply_cutout(img, size=16, p=0.5), x)
        return x, y
      ds = ds.map(augment_with_cutout, num_parallel_calls=tf.data.AUTOTUNE)

  ds = ds.repeat().prefetch(tf.data.AUTOTUNE)

  # ---------- Generator ----------
  def generator():
    for x_batch, y_batch in ds:
      x_batch = x_batch.numpy().astype(jnp.float32) / 255.0
      if dataset == 'mnist' or dataset == 'truncated_mnist':
        x_batch = (x_batch - mean) / std
      else:
        x_batch = (x_batch - mean[None, None, None, :]) / std[None, None, None, :]
      yield jnp.array(x_batch, dtype=jnp.float32), jnp.array(y_batch.numpy(), dtype=jnp.int32)

  return generator(), info


def compute_batch_accuracy(state, x_batch, y_batch):
  """
  Computes accuracy for a single batch.

  Args:
      params: Model parameters.
      model: Flax model.
      x_batch: Input data for the batch.
      y_batch: Ground truth labels for the batch.

  Returns:
      Accuracy for the batch as a percentage (float).
      And Loss
  """
  # Forward pass to compute logits
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  log_probs = state.apply_inf_fn(variables, x_batch, mutable=False)  # shape [batch_size, num_classes]

  # Predicted class (argmax of logits)
  predictions = jnp.argmax(jnp.exp(log_probs), axis=1)

  # Compare predictions with ground truth
  correct_predictions = jnp.sum(predictions == y_batch)

  # Compute accuracy
  accuracy = (correct_predictions / y_batch.shape[0]) * 100
  return accuracy, cross_entropy_loss(log_probs, y_batch)

# def compute_accuracy_and_loss(state, dataloader):
#   """
#   Computes accuracy for the given model parameters and dataloader.
#
#   Args:
#       params: Model parameters.
#       model: Flax model.
#       dataloader: DataLoader for the dataset.
#
#   Returns:
#       Accuracy as a percentage (float).
#       Averaged loss
#   """
#   # If this is a custom loader, reset it to ensure we get a fresh epoch
#   if hasattr(dataloader, "reset"):
#     dataloader.reset()
#   init_batch = next(dataloader)
#   batch_axis = infer_batch_layout(init_batch)["batch_axis"]
#   x_batch, y_batch = init_batch
#   batch_size = x_batch.shape[batch_axis]
#   pred_size = y_batch.shape[0]
#   acc, loss = compute_batch_accuracy(state, x_batch, y_batch)
#   correct_predictions = acc * pred_size
#   running_loss = loss * pred_size
#   final_pred_size = pred_size
#   total_samples = batch_size
#
#   for x_batch, y_batch in dataloader:
#     # Compute model predictions
#     pred_size = y_batch.shape[0]
#     acc, loss = compute_batch_accuracy(state, x_batch, y_batch)
#     correct_predictions += acc * pred_size
#     running_loss += loss * pred_size
#     total_samples += batch_size
#     final_pred_size += pred_size
#     if total_samples >= 10000:
#       break
#
#   # Compute accuracy as a percentage
#   accuracy = (correct_predictions / final_pred_size)
#   return accuracy, running_loss / final_pred_size

def compute_accuracy_and_loss(state, dataloader, num_samples: int):
  """
  Evaluate on ~num_samples examples from dataloader (works with infinite repeat()).
  """
  if hasattr(dataloader, "reset"):
    dataloader.reset()

  correct_predictions = 0.0
  running_loss = 0.0
  seen = 0

  for x_batch, y_batch in dataloader:
    pred_size = y_batch.shape[0]          # robust: counts labels actually present
    acc, loss = compute_batch_accuracy(state, x_batch, y_batch)

    correct_predictions += acc * pred_size
    running_loss += loss * pred_size
    seen += pred_size

    if seen >= num_samples:
      break

  accuracy = correct_predictions / seen
  avg_loss = running_loss / seen
  return accuracy, avg_loss


def compute_accuracy_and_loss_with_hists(
    state,
    dataloader,
    num_samples: int,
    run,                      # Aim Run
    *,
    step: int,
    prefix: str = "eval",     # e.g. "train_eval" or "test"
    context: dict | None = None,
    bin_count: int = 50,
    max_points: int | None = 200_000,   # optional subsample cap (recommended for ImageNet)
    rng_seed: int = 0,
):
  """
  Evaluate on ~num_samples examples from dataloader (works with repeat()) and track:
    1) Distribution of p(correct class)
    2) Distribution of per-sample NLL

  Returns:
    (accuracy_percent, avg_loss)
  """
  import numpy as np
  import jax
  import jax.numpy as jnp
  from aim import Distribution

  if hasattr(dataloader, "reset"):
    dataloader.reset()

  correct_sum = 0.0
  loss_sum = 0.0
  seen = 0

  # Optional reservoir sampling so you don't log millions of points
  rng = np.random.default_rng(rng_seed)

  p_correct_samples = []
  nll_samples = []
  if max_points is not None:
    cap = int(max_points)
    stream_n = 0  # total points seen in stream for reservoir sampling

    def _reservoir_add(arr_p: np.ndarray, arr_nll: np.ndarray):
      nonlocal stream_n, p_correct_samples, nll_samples
      for p, nll in zip(arr_p.tolist(), arr_nll.tolist()):
        stream_n += 1
        if len(p_correct_samples) < cap:
          p_correct_samples.append(p)
          nll_samples.append(nll)
        else:
          j = rng.integers(0, stream_n)
          if j < cap:
            p_correct_samples[j] = p
            nll_samples[j] = nll
  else:
    def _reservoir_add(arr_p: np.ndarray, arr_nll: np.ndarray):
      p_correct_samples.extend(arr_p.tolist())
      nll_samples.extend(arr_nll.tolist())

  for x_batch, y_batch in dataloader:
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    log_probs = state.apply_inf_fn(variables, x_batch, mutable=False)  # [B, C]

    # Preds (your version used argmax(exp(log_probs)); argmax(log_probs) is equivalent)
    preds = jnp.argmax(log_probs, axis=1)

    # Per-sample correctness (0/1)
    correct = (preds == y_batch).astype(jnp.float32)

    # p(correct class) and per-sample NLL
    idx = jnp.arange(y_batch.shape[0])
    lp_true = log_probs[idx, y_batch]                 # log p(y|x)
    p_true = jnp.exp(lp_true)                         # p(y|x) in [0,1]
    nll = -lp_true                                    # NLL >= 0

    bs = int(y_batch.shape[0])
    correct_sum += float(jnp.sum(correct))
    loss_sum += float(jnp.sum(nll))
    seen += bs

    # move to host for Aim histogram
    _reservoir_add(np.asarray(p_true), np.asarray(nll))

    if seen >= num_samples:
      break

  acc_percent = (correct_sum / seen) * 100.0
  avg_loss = loss_sum / seen

  # Track histograms
  run.track(
    Distribution(p_correct_samples, bin_count=bin_count),
    name=f"{prefix}/p_correct_hist",
    step=step,
    context=context,
  )
  run.track(
    Distribution(nll_samples, bin_count=bin_count),
    name=f"{prefix}/nll_hist",
    step=step,
    context=context,
  )

  return acc_percent, avg_loss

 # Prepare the train state for the model parameters
  # (Using Flax's train_state for convenience)
class TrainState(train_state.TrainState):
  apply_inf_fn: Callable = struct.field(pytree_node=False)
  batch_stats: Any
  gbar: Any
  g_last: Any

  def apply_gradients(self, *, grads, normalize_conv_params=False, **kwargs):
    """Updates ``step``, ``params``, ``opt_state`` and ``**kwargs`` in return value.

    Note that internally this function calls ``.tx.update()`` followed by a call
    to ``optax.apply_updates()`` to update ``params`` and ``opt_state``.

    Args:
      grads: Gradients that have the same pytree structure as ``.params``.
      normalize_conv_params: Whether to normalize conv kernel params before applying updates.
      **kwargs: Additional dataclass attributes that should be ``.replace()``-ed.

    Returns:
      An updated instance of ``self`` with ``step`` incremented by one, ``params``
      and ``opt_state`` updated by applying ``grads``, and additional attributes
      replaced as specified by ``kwargs``.
    """
    if OVERWRITE_WITH_GRADIENT in grads:
      grads_with_opt = grads['params']
      params_with_opt = self.params['params']
    else:
      grads_with_opt = grads
      params_with_opt = self.params

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt
    )
    if normalize_conv_params:
      params_with_opt = normalize_conv_params_l2(params_with_opt)
    new_params_with_opt = optax.apply_updates(params_with_opt, updates)

    # As implied by the OWG name, the gradients are used directly to update the
    # parameters.
    if OVERWRITE_WITH_GRADIENT in grads:
      new_params = {
        'params': new_params_with_opt,
        OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
      }
    else:
      new_params = new_params_with_opt
    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )

  def apply_gradients_and_precond(self, *, grads, precond_apply, normalize_conv_params=False, **kwargs):
    """Same as original apply_gradients from Flax, but applies preconditiner on update instead of gradient
    (i.e. after applying momentum for example)
    """
    # From original -->
    if OVERWRITE_WITH_GRADIENT in grads:
      grads_with_opt = grads['params']
      params_with_opt = self.params['params']
    else:
      grads_with_opt = grads
      params_with_opt = self.params

    updates, new_opt_state = self.tx.update(
      grads_with_opt, self.opt_state, params_with_opt
    )
    # <-- until here
    if normalize_conv_params:
      params_with_opt = normalize_conv_params_l2(params_with_opt)
    new_params_with_opt = optax.apply_updates(params_with_opt, precond_apply(updates))
    # And then again from original -->

    # As implied by the OWG name, the gradients are used directly to update the
    # parameters.
    if OVERWRITE_WITH_GRADIENT in grads:
      new_params = {
        'params': new_params_with_opt,
        OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
      }
    else:
      new_params = new_params_with_opt
    return self.replace(
      step=self.step + 1,
      params=new_params,
      opt_state=new_opt_state,
      **kwargs,
    )

  @classmethod
  def create(cls, *, apply_fn, params, tx, opt_state=None, **kwargs):
    """Creates a new instance with ``step=0`` and initialized ``opt_state``."""
    # We exclude OWG params when present because they do not need opt states.
    params_with_opt = (
      params['params'] if OVERWRITE_WITH_GRADIENT in params else params
    )
    if not opt_state:
      opt_state = tx.init(params_with_opt)
    return cls(
      step=0,
      apply_fn=apply_fn,
      params=params,
      tx=tx,
      opt_state=opt_state,
      **kwargs,
    )


def mask_from_flat_keys(pytree, substrings, delimiter="/"):
  """
  Build a boolean mask pytree with the same structure as `pytree`.
  A leaf is False iff any substring in `substrings` appears in its flattened key path.
  """

  # --- Input validation for `substrings` ---
  if isinstance(substrings, (str, bytes)):
    raise TypeError(
      "substrings must be a sequence of strings (e.g., ['bias', 'LayerNorm']), "
      "not a single string."
    )
  if not isinstance(substrings, Sequence):
    raise TypeError(
      f"substrings must be a sequence of strings; got {type(substrings).__name__}."
    )
  if not all(isinstance(s, str) for s in substrings):
    bad = [type(s).__name__ for s in substrings if not isinstance(s, str)]
    raise TypeError(
      f"all items in `substrings` must be str; got non-str types: {set(bad)}."
    )

  subs = tuple(substrings)

  flat_dict = flatten_dict(pytree)
  new_flat_dict = {}
  for k, v in flat_dict.items():
    new_flat_dict[k] = not any(n in s for s in k for n in subs)
  return unflatten_dict(new_flat_dict)


def infer_batch_layout(batch): # TODO: potentially not robust, might be preferable to add a batch_axis arg to dataset configs
    """
    Infer where the batch dimension lives and how to treat targets.

    Assumes `batch` is (x, y) or a pytree whose first two leaves are (x, y)
    with either:
      - CV-style:   x: [B, ...], y: [B] or [B, ...]
      - Text-style: x: [T, B],   y: [T*B]

    Returns:
      layout = {
        "mode": "cv" or "text",
        "batch_axis": int,          # axis of batch dim in x
        "T": int | None,            # sequence length for text mode
      }
    """
    leaves = jax.tree_util.tree_leaves(batch)
    assert len(leaves) >= 2, "Expected at least (x, y) in batch pytree"
    x, y = leaves[0], leaves[1]

    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)
    if not isinstance(y, jnp.ndarray):
        y = jnp.asarray(y)

    # Default assumption: CV-like
    layout = {"mode": "cv", "batch_axis": 0, "T": None}

    # Text-mode pattern: x: [T, B], y: [T*B]
    if x.ndim == 2 and y.ndim == 1:
        T0, B1 = x.shape
        # If y is T*B and != B (avoid ambiguous case where T==1)
        if y.shape[0] % B1 == 0 and y.shape[0] != T0:
            T = y.shape[0] // B1
            layout = {"mode": "text", "batch_axis": 1, "T": int(T)}
            return layout

    # CV fallback: batch axis is 0 (x: [B, ...], y: [B] or [B, ...])
    return layout


# ---------------------------------------------------------
# 1) Jit helpers for train_step
# ---------------------------------------------------------
def next_accumulated_batches(train_dataloader, acc_steps):
  xs, ys = [], []
  for _ in range(acc_steps):
    x, y = next(train_dataloader)
    xs.append(x)
    ys.append(y)

  # If x,y are already jax arrays, jnp.stack is fine.
  # If they're numpy, jnp.asarray will transfer to device once (good).
  x_acc = jnp.stack([jnp.asarray(x) for x in xs], axis=0)
  y_acc = jnp.stack([jnp.asarray(y) for y in ys], axis=0)
  return x_acc, y_acc


# ---------------------------------------------------------
# ASAM (Adaptive SAM) helpers
# ---------------------------------------------------------
def tree_zeros_like(tree):
  return jax.tree_map(jnp.zeros_like, tree)

def tree_add(a, b):
  return jax.tree_map(jnp.add, a, b)

def tree_sub(a, b):
  return jax.tree_map(jnp.subtract, a, b)

def tree_mul_scalar(tree, s):
  return jax.tree_map(lambda x: x * s, tree)

def tree_dot(a, b):
  # sum over all leaves of sum(a_leaf * b_leaf)
  leaves = jax.tree_util.tree_leaves(jax.tree_map(lambda x, y: jnp.sum(x * y), a, b))
  return jnp.sum(jnp.stack(leaves)) if leaves else jnp.array(0.0, dtype=jnp.float32)

def tree_l2_norm(tree, eps=1e-12):
  return jnp.sqrt(tree_dot(tree, tree) + eps)

def tree_normalize(tree, eps=1e-12):
  n = tree_l2_norm(tree, eps)
  return tree_mul_scalar(tree, 1.0 / n)

def tree_dot_subtree(x_sub, y_sub):
  leaves = jax.tree_util.tree_leaves(
    jax.tree_map(lambda x, y: jnp.sum(x * y), x_sub, y_sub)
  )
  return jnp.sum(jnp.stack(leaves)) if leaves else jnp.array(0.0, dtype=jnp.float32)

def tree_dot_per_layer(a, b):
  return jax.tree_map(tree_dot_subtree, a, b)

def tree_l2_norm_per_layer(tree, eps=1e-12):
  return jax.tree_map(lambda d: jnp.sqrt(d + eps), tree_dot_per_layer(tree, tree))

def tree_normalize_per_layer(tree, eps=1e-12):
  norms = tree_l2_norm_per_layer(tree, eps)
  return jax.tree_map(lambda sub, n: jax.tree_map(lambda x: x / n, sub), tree, norms)

def make_perturbation_from_vector(
    *,
    precond_blocks,
    vector,
    precond_apply_fn,
    rho: float,
    mode: str,
    eps: float = 1e-12,
):
  """Build a perturbation tree from an already chosen perturbation vector."""
  vector = jax.lax.stop_gradient(vector)

  if mode == "ema_grad":
    transformed = vector
  elif mode == "ema_precond_grad":
    transformed = precond_apply_fn(precond_blocks, vector)
  elif mode == "ema_direction":
    transformed = precond_apply_fn(precond_blocks, vector)
    transformed = tree_normalize(transformed, eps)
  else:
    raise ValueError(f"Unknown perturb_mode: {mode}")

  # sqrt(v^T P^T P v). Less principled than sqrt(v^T P v), but more stable.
  denom = jnp.sqrt(tree_dot(transformed, transformed) + eps)
  direction = tree_mul_scalar(transformed, 1.0 / denom)
  return tree_mul_scalar(direction, rho)


def make_perturbation_from_grad(
    *,
    precond_blocks,
    grad,
    precond_apply_fn,
    rho: float,
    mode: str,
    eps: float = 1e-12,
):
  """Thin wrapper for SAM-style perturbations built directly from a gradient."""
  return make_perturbation_from_vector(
    precond_blocks=precond_blocks,
    vector=grad,
    precond_apply_fn=precond_apply_fn,
    rho=rho,
    mode=mode,
    eps=eps,
  )


def make_perturbation_from_noise(
    precond_blocks,
    g_last,
    g_bar,
    precond_apply_fn,
    rho: float,
    mode: str,
    eps: float = 1e-12,
):
  """
  Returns epsilon pytree to add to params, where direction is aligned with
  the "noise proxy" v = g_last - g_bar.
  """
  g_last = jax.lax.stop_gradient(g_last)
  g_bar  = jax.lax.stop_gradient(g_bar)
  v = tree_sub(g_last, g_bar)
  return make_perturbation_from_vector(
    precond_blocks=precond_blocks,
    vector=v,
    precond_apply_fn=precond_apply_fn,
    rho=rho,
    mode=mode,
    eps=eps,
  )


def update_gbar(
    g_bar,
    mean_grads_pert,
    precond_blocks,
    precond_apply_fn,
    beta: float,
    mode: str,
    eps: float = 1e-12,
):
  """
  Keep the same 3 modes as before, but note the training step now uses g_last - g_bar
  to build perturbations.

  mean_grads_pert = ∇L(θ + ε) averaged across accumulation steps.
  """
  # if mode == "ema_grad":
  #   target = mean_grads_pert
  #
  # elif mode == "ema_precond_grad":
  #   target = precond_apply_fn(precond_blocks, mean_grads_pert)  # P g^{pert}
  #
  # elif mode == "ema_direction":
  #   Pg = precond_apply_fn(precond_blocks, mean_grads_pert)
  #   target = tree_normalize(Pg, eps)  # unit direction in P-space
  #
  # else:
  #   raise ValueError(f"Unknown perturb_mode: {mode}")

  target = mean_grads_pert

  new_gbar = tree_add(
    tree_mul_scalar(g_bar, beta),
    tree_mul_scalar(target, 1.0 - beta),
  )

  if mode == "ema_direction":
    new_gbar = tree_normalize(new_gbar, eps)

  return jax.lax.stop_gradient(new_gbar)


#################################
# Checkpointing utils
#################################
# Preemption handling on cluster
class RunState(TypedDict):  # Taken from https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
  """Typed dictionary containing the state of the training run which is saved at each epoch.

  Using type hints helps prevent bugs and makes your code easier to read for both humans and
  machines (e.g. Copilot). This leads to less time spent debugging and better code suggestions.
  """

  epoch: int
  training_step: int
  model_dir: str  # Parent dir containing trainstate and preconditioner pytrees in separate children dir
  aim_hash: Optional[str]  # Unique hash identifying experiment in aim (logger)
  slurm_jobid: str  # Unique experiment identifier attributed by SLURM
  exp_name: str
  dropout_key: Optional[jax.random.PRNGKey]
  best_accuracy: float  # Best accuracy so far
  training_time: Optional[Any]  # Total training time for the run


def load_run_state(checkpoint_dir: Path) -> Optional[
  RunState]:  # Taken from https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
  """Loads the latest checkpoint if possible, otherwise returns `None`."""
  checkpoint_file = checkpoint_dir / "checkpoint_run_state.pkl"
  restart_count = int(os.environ.get("SLURM_RESTART_COUNT", 0))
  if restart_count:
    print(f"NOTE: This job has been restarted {restart_count} times by SLURM.")

  if not checkpoint_file.exists():
    print(f"No checkpoint found in checkpoints dir ({checkpoint_dir}).")
    if restart_count:
      raise RuntimeWarning(
        f"This job has been restarted {restart_count} times by SLURM, but no "
        "checkpoint was found! This either means that your checkpointing code is "
        "broken, or that the job did not reach the checkpointing portion of your "
        "training loop."
      )
    return None

  with open(checkpoint_file, "rb") as f:
    checkpoint_state = pickle.load(f)

  print(f"Resuming from the checkpoint file at {checkpoint_file}:")
  print(checkpoint_state)
  print()
  state: RunState = checkpoint_state  # type: ignore
  return state


# def save_pytree_state(ckpt_dir: str, state) -> None:
#   # Save the numpy arrays (parameters) to disk
#   with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
#     for x in jax.tree_util.tree_leaves(state):
#       jnp.save(f, x, allow_pickle=True)
#
#   # Save the structure of the state tree
#   tree_struct = jax.tree_map(lambda t: 0, state)
#   with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
#     pickle.dump(tree_struct, f)

def save_pytree_state(ckpt_dir: str, pytree) -> None:
  os.makedirs(ckpt_dir, exist_ok=True)

  # Flatten the pytree into leaves and treedef
  leaves, treedef = jax.tree_util.tree_flatten(pytree)

  # Convert leaves to numpy arrays explicitly and save
  with open(os.path.join(ckpt_dir, "arrays.npz"), "wb") as f:
    np.savez(f, *[np.array(leaf) for leaf in leaves])

  # Save the treedef structure separately
  with open(os.path.join(ckpt_dir, "treedef.pkl"), "wb") as f:
    pickle.dump(treedef, f)


# def restore_pytree_state(ckpt_dir, verbose=False):
#   # Load the structure of the state tree
#   with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
#     tree_struct = pickle.load(f)
#
#   if verbose:
#     print(jax.tree_map(jnp.shape, tree_struct))
#
#   # Load the flat state (parameters) from disk
#   leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
#   with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
#     flat_state = [jnp.load(f, allow_pickle=True) for _ in leaves]
#
#   # Reconstruct the state tree from its structure and parameters
#   return jax.tree_util.tree_unflatten(treedef, flat_state)

def restore_pytree_state(ckpt_dir, verbose=False):
  # Load treedef
  with open(os.path.join(ckpt_dir, "treedef.pkl"), "rb") as f:
    treedef = pickle.load(f)

  # Load arrays
  with np.load(os.path.join(ckpt_dir, "arrays.npz"), allow_pickle=False) as data:
    leaves = [data[key] for key in data.files]

  restored_tree = jax.tree_util.tree_unflatten(treedef, leaves)

  if verbose:
    print(jax.tree_map(lambda x: x.shape, restored_tree))

  return restored_tree


def save_trainstate_and_precond(parent_dir: str, trainstate, preconditioner_blocks) -> None:
  # Create directories for params, state, and opt_state
  trainstate_dir = os.path.join(parent_dir, "trainstate")
  params_dirs = os.path.join(trainstate_dir, "params")
  gbar_dirs = os.path.join(trainstate_dir, "gbar")
  g_last_dirs = os.path.join(trainstate_dir, "g_last")
  opt_state_dir = os.path.join(trainstate_dir, "opt_state")
  batch_stats_dir = os.path.join(trainstate_dir, "batch_stats")

  precond_dir = os.path.join(parent_dir, "preconditioner")
  os.makedirs(trainstate_dir, exist_ok=True)
  os.makedirs(params_dirs, exist_ok=True)
  os.makedirs(gbar_dirs, exist_ok=True)
  os.makedirs(g_last_dirs, exist_ok=True)
  os.makedirs(opt_state_dir, exist_ok=True)
  os.makedirs(batch_stats_dir, exist_ok=True)
  os.makedirs(precond_dir, exist_ok=True)

  # Use the existing save function
  save_pytree_state(params_dirs, trainstate.params)
  save_pytree_state(gbar_dirs, trainstate.gbar)
  save_pytree_state(g_last_dirs, trainstate.g_last)
  save_pytree_state(opt_state_dir, trainstate.opt_state)
  save_pytree_state(batch_stats_dir, trainstate.batch_stats)
  save_pytree_state(precond_dir, preconditioner_blocks)


def restore_trainstate_and_precond(parent_dir: str):
  # Directories for params, state, and opt_state
  trainstate_dir = os.path.join(parent_dir, "trainstate")
  precond_dir = os.path.join(parent_dir, "preconditioner")
  params_dirs = os.path.join(trainstate_dir, "params")
  gbar_dirs = os.path.join(trainstate_dir, "gbar")
  g_last_dirs = os.path.join(trainstate_dir, "g_last")
  opt_state_dir = os.path.join(trainstate_dir, "opt_state")
  batch_stats_dir = os.path.join(trainstate_dir, "batch_stats")

  # Use the existing restore function
  restored_params = restore_pytree_state(params_dirs)
  restored_gbar = restore_pytree_state(gbar_dirs)
  restored_g_last = restore_pytree_state(g_last_dirs)
  restored_opt_state = restore_pytree_state(opt_state_dir)
  restored_batch_stats = restore_pytree_state(batch_stats_dir)
  restored_preconditioner_blocks = restore_pytree_state(precond_dir)

  return {"params": restored_params, "gbar":restored_gbar, "g_last":restored_g_last, "opt_state": restored_opt_state, "batch_stats":restored_batch_stats}, restored_preconditioner_blocks

def checkpoint_exp(run_state: RunState, trainstate, precond_blocks, curr_epoch: int, curr_step: int,
                   dropout_key, best_acc, training_time):
  run_state["epoch"] = curr_epoch
  run_state["training_step"] = curr_step
  run_state["dropout_key"] = dropout_key
  run_state["best_accuracy"] = best_acc
  run_state["training_time"] = training_time

  with open(os.path.join(run_state["model_dir"], "checkpoint_run_state.pkl"), "wb") as f:
    pickle.dump(run_state, f)

  # Update weights
  save_trainstate_and_precond(run_state["model_dir"], trainstate, precond_blocks)


def signal_handler(signum: int, frame: Optional[
  FrameType]):  # Taken from: https://docs.mila.quebec/examples/good_practices/checkpointing/index.html
  """Called before the job gets pre-empted or reaches the time-limit.

  This should run quickly. Performing a full checkpoint here mid-epoch is not recommended.
  """
  signal_enum = signal.Signals(signum)
  print(f"Job received a {signal_enum.name} signal!")


  ##################################
  # Hyperparam utils
  ##################################
def cosine_annealing_schedule_per_epoch( #TODO: rename, not per epoch anymore
        base_lr: float,
        total_epochs: int,
        steps_per_epoch: float,
        t_max: int,
        eta_min: float = 0.0,
        cycle: bool = True, # Whether we cycle through lr after t_max is reached, or fix lr to eta_min.
) -> optax.Schedule:
  """
  Optax-compatible schedule that mimics PyTorch's CosineAnnealingLR
  behavior, where the learning rate is updated per epoch.

  Args:
      base_lr: Initial learning rate (maximum value).
      t_max: Total number of epochs until eta_min is reached.
      steps_per_epoch: Number of steps (batches) in one epoch.
      eta_min: Final learning rate value (default: 0.0).

  Returns:
      A function that maps step counts to learning rates.
  """
  if t_max <= 0:
    raise ValueError(f"T_max must be positive, got {t_max}")
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
  step_max = steps_per_epoch * t_max

  if cycle:
    def schedule(t):
      # t = jnp.floor_divide(step_count, steps_per_epoch).astype(jnp.float32)
      # t = jnp.minimum(t, t_max)
      # cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * t / t_max))
      cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * t / step_max))
      return eta_min + (base_lr - eta_min) * cosine_decay
  else:
    def schedule(t):
      t = jnp.minimum(t, step_max)
      cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * t / step_max))
      return eta_min + (base_lr - eta_min) * cosine_decay

  return schedule


def warmup_cosine_annealing_schedule(
    base_lr: float,
    total_epochs: int,
    steps_per_epoch: float,
    init_lr: float,
    warmup_epochs: int,
    t_max: int,
    eta_min: float = 0.0,):
  if t_max <= 0:
    raise ValueError(f"T_max must be positive, got {t_max}")
  if steps_per_epoch <= 0:
    raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
  warmup_steps = int(warmup_epochs * steps_per_epoch)
  step_max = int(steps_per_epoch * t_max)

  return optax.schedules.warmup_cosine_decay_schedule(init_value=init_lr, peak_value=base_lr,
                                                      warmup_steps=warmup_steps, decay_steps=step_max, end_value=eta_min)


def step_warmup(
        base_lr: float,
        total_epochs: int,
        steps_per_epoch: float,
        warmup_ratio: float=1e-5,
        init_value: float=0.0,
) -> optax.Schedule:
  """ A simple linear warmup at the beginning, implemented for the small grokking experiments"""
  warmup_steps = total_epochs * steps_per_epoch * warmup_ratio
  diff = base_lr - init_value
  return lambda s: init_value + diff * jnp.minimum(s / warmup_steps, 1)


def linear_schedule(
        base_lr: float,
        total_epochs: int,
        steps_per_epoch: float,
        decay_factor: float, # by how much we reduce the lr over transition steps
        transition_epochs: int,
        transition_begin: int,
) -> optax.Schedule:
  transition_steps = int(steps_per_epoch * transition_epochs)
  transition_begin = int(steps_per_epoch * transition_begin)
  end_value = decay_factor * base_lr
  return optax.linear_schedule(init_value=base_lr, end_value=end_value, transition_steps=transition_steps, transition_begin=transition_begin)


def piecewise_constant_schedule(
        base_lr: float,
        total_epochs: int,
        steps_per_epoch: float,
        boundaries: dict # of the form {epoch: scale}
) -> optax.Schedule:
  optax_boundaries = {steps_per_epoch*key:value for key,value in boundaries.items()}
  return optax.piecewise_constant_schedule(init_value=base_lr, boundaries_and_scales=optax_boundaries)

def warmup_piecewise_decay_schedule(
        base_lr: float,
        total_epochs: int,
        steps_per_epoch: float,
        epoch_decay_bounds: List, # of epochs, when scaling_factor (decay) is applied
        scaling_factor: float,
        warmup_ratio: float = 0.05
) -> optax.Schedule:
  """Linear warmup followed by piecewise decay
  """
  training_steps = int(total_epochs * steps_per_epoch)
  warmup_steps = int(warmup_ratio * training_steps)  # warmup is done for 5% of training_steps
  _step_decay_bounds = [steps_per_epoch * lr_decay_step for lr_decay_step in epoch_decay_bounds]
  bound_dict = {i - warmup_steps: scaling_factor for i in _step_decay_bounds}
  schedules = [
    optax.linear_schedule(
      init_value=1e-6,
      end_value=base_lr,
      transition_steps=warmup_steps),
    optax.piecewise_constant_schedule(
      init_value=base_lr,
      boundaries_and_scales=bound_dict)]
  return optax.join_schedules(schedules, [warmup_steps])


##################################
# Measuring matrix asymmetry
##################################
def _check_square(A: jnp.ndarray):
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(
            f"Expected a square matrix, got shape {A.shape}."
        )


@jax.jit
def rel_skew_energy_fro_vs_sym(A: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Relative skew-energy using Frobenius norms, reporting skew vs symmetric part:
        ||S||_F / (||H||_F + eps)
    where H = (A + A^T)/2, S = (A - A^T)/2.
    """
    H = 0.5 * (A + A.T)
    S = 0.5 * (A - A.T)
    return jnp.linalg.norm(S, ord="fro") / (jnp.linalg.norm(H, ord="fro") + eps)


def _spectral_norm_power_iteration(
    M: jnp.ndarray, iters: int = 25, eps: float = 1e-12
) -> jnp.ndarray:
    """
    Estimates ||M||_2 (largest singular value) via power iteration on M^T M.
    Deterministic initialization (all-ones vector) to avoid RNG in the signature.
    """
    n = M.shape[1]
    v0 = jnp.ones((n,), dtype=M.dtype)
    v0 = v0 / (jnp.linalg.norm(v0, ord=2) + eps)

    def body(v, _):
        v = M.T @ (M @ v)
        v = v / (jnp.linalg.norm(v, ord=2) + eps)
        return v, None

    v, _ = jax.lax.scan(body, v0, xs=None, length=iters)
    return jnp.linalg.norm(M @ v, ord=2)


@jax.jit
def rel_skew_operator_norm(
    A: jnp.ndarray, iters: int = 25, eps: float = 1e-12
) -> jnp.ndarray:
    """
    Relative skew measured in operator (induced 2-) norm:
        ||A - A^T||_2 / (||A||_2 + eps)
    with ||·||_2 estimated by power iteration.
    """
    D = A - A.T
    num = _spectral_norm_power_iteration(D, iters=iters, eps=eps)
    den = _spectral_norm_power_iteration(A, iters=iters, eps=eps)
    return num / (den + eps)


def report_skews(A: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  _check_square(A)

  frob_skew = rel_skew_energy_fro_vs_sym(A)
  spectral_skew = rel_skew_operator_norm(A)
  return frob_skew, spectral_skew


def get_per_layer_skews(precond):
  skews_dict = {}
  for _layer, _block in precond.items():
    layer_skews = []
    for leaf in jax.tree_leaves(_block):
      if leaf.ndim == 2:
        layer_skews.append(report_skews(leaf))
    skews_dict[_layer] = layer_skews

  return skews_dict

def average_skews(
        skews_dict: Dict[str, List[Tuple[jnp.ndarray, jnp.ndarray]]],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
  Averages Frobenius and spectral skews across all leaves in skews_dict.

  Args:
    skews_dict: {layer: [(frob_skew, spectral_skew), ...], ...}
    eps: numerical guard in case no leaves are present.

  Returns:
    (avg_frob_skew, avg_spectral_skew)
  """
  frob_vals = []
  spectral_vals = []

  for layer_skews in skews_dict.values():
    for frob_skew, spectral_skew in layer_skews:
      frob_vals.append(frob_skew)
      spectral_vals.append(spectral_skew)

  if len(frob_vals) == 0:
    # No valid leaves: return zeros with correct dtype semantics
    zero = jnp.array(0.0)
    return zero, zero

  frob_stack = jnp.stack(frob_vals)
  spectral_stack = jnp.stack(spectral_vals)

  avg_frob = jnp.mean(frob_stack)
  avg_spectral = jnp.mean(spectral_stack)

  return avg_frob, avg_spectral

##################################
# "Trick" utils
##################################
def normalize_conv_params_l2(params):
    """
    Normalize parameter tensors to have L2 norm = sqrt(N) if they have >= 4 dimension.
    Leaves with only 1 dimension (e.g. biases) are ignored.

    Args:
        params (PyTree): A nested dict of model parameters.

    Returns:
        PyTree with the same structure and normalized parameters.
    """
    def maybe_normalize_leaf(p):
        if p.ndim < 4:
            return p
        norm = jnp.linalg.norm(p)
        scale = jnp.sqrt(p.size) / jnp.maximum(norm, 1e-8)
        return p * scale

    return jax.tree_util.tree_map(maybe_normalize_leaf, params)


##################################
# Varia
##################################
def _deep_copy_pytree(tree):
  # Ensures every array leaf gets a fresh device buffer (no aliasing).
  def _copy_leaf(x):
    # jnp.array(x, copy=True) is the most explicit; .copy() also works on arrays.
    return jnp.array(x, copy=True) if isinstance(x, jnp.ndarray) else x
  return jax.tree_map(_copy_leaf, tree)
