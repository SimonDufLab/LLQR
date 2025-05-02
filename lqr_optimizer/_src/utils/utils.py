""" Various utilities functions for LQR optimization"""
import time
import os
import pickle
import signal
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import flax.linen as nn
from flax.linen import Sequential
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from typing import List, Tuple, Any, Dict, Optional, TypedDict, Callable
from types import FrameType
from pathlib import Path

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


##################################
# Loading utils
##################################
def load_main_optimizer(cfg):
  if cfg.main_optimizer == "polyak":
    model_optimizer = optax.sgd(learning_rate=cfg.learning_rate, momentum=cfg.momentum)
  elif cfg.main_optimizer == "adam":
    model_optimizer = optax.adam(learning_rate=cfg.learning_rate)
  elif cfg.main_optimizer == "sgd":
    model_optimizer = optax.sgd(learning_rate=cfg.learning_rate)
  else:
    raise ValueError("Unknown main optimizer")
  return model_optimizer


def load_precond_optimizer(cfg):
  if cfg.precond_solver == "adam":
    optax_solver_for_precond = optax.adam(cfg.precond_lr)
  elif cfg.precond_solver == "momentum":
    optax_solver_for_precond = optax.sgd(cfg.precond_lr, momentum=cfg.momentum)
  elif cfg.precond_solver == "sgd":
    optax_solver_for_precond = optax.sgd(cfg.precond_lr)
  else:
    raise ValueError("Unknown precond optimizer")
  return optax_solver_for_precond


##################################
# Training utils
##################################
# Simple cross-entropy loss for classification
def cross_entropy_loss(log_probs, y): #TODO only with respect to output
  """
  log_probs: network outputs, the per-class log probabilities
  y: integer class labels
  """

  # Cross-entropy using negative log-likelihood
  nll = -jnp.mean(jnp.sum(jax.nn.one_hot(y, log_probs.shape[-1]) * log_probs, axis=-1))
  return nll

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


def prepare_dataloader(batch_size=128, train=True, dataset='mnist', augment_dataset=False):
  """
  Creates a generator that yields (x, y) from the specified dataset (MNIST, CIFAR-10, or CIFAR-100).
  Applies specified data augmentation to CIFAR datasets if augment_dataset=True.
  Returns the generator along with the number of classes in the dataset.
  """
  if dataset == 'mnist' or dataset == 'truncated_mnist':
    ds_name = 'mnist'
    mean = 0.1307
    std = 0.3081
    num_classes = 10
  elif dataset == 'cifar-10':
    ds_name = 'cifar10'
    mean = jnp.array([0.4914, 0.4822, 0.4465])
    std = jnp.array([0.2470, 0.2435, 0.2616])
    num_classes = 10
  elif dataset == 'cifar-100':
    ds_name = 'cifar100'
    mean = jnp.array([0.5071, 0.4867, 0.4408])
    std = jnp.array([0.2675, 0.2565, 0.2761])
    num_classes = 100
  else:
    raise ValueError("Unsupported dataset. Choose either 'mnist', 'truncated_mnist', 'cifar-10', or 'cifar-100'")

  ds, info = tfds.load(ds_name, split='train' if train else 'test', as_supervised=True, with_info=True)

  if dataset == 'truncated_mnist' and train:
    # Shuffle and take a subset of 10,000 examples
    ds = ds.shuffle(buffer_size=info.splits['train'].num_examples, seed=0)
    ds = ds.take(10_000)

  ds_size = int(ds.cardinality())
  ds = ds.cache()
  ds = ds.shuffle(ds_size, seed=0, reshuffle_each_iteration=True)
  ds = ds.batch(batch_size)

  if augment_dataset and train and dataset in ['cifar-10', 'cifar-100']:
    ReflectionPadding2D = tf.keras.layers.Lambda(lambda x: tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], 'REFLECT'))
    augmentation_pipeline = tf.keras.Sequential([
      ReflectionPadding2D,
      tf.keras.layers.RandomCrop(height=32, width=32),
      tf.keras.layers.RandomFlip('horizontal')
    ])
    ds = ds.map(lambda x, y: (augmentation_pipeline(x), y), num_parallel_calls=tf.data.AUTOTUNE)

  ds = ds.repeat().prefetch(tf.data.AUTOTUNE)

  def generator():
    for x_batch, y_batch in ds:
      x_batch = x_batch.numpy()
      y_batch = y_batch.numpy()

      x_batch = x_batch.astype(jnp.float32) / 255.0
      if dataset == 'mnist':
        x_batch = (x_batch - mean) / std
      else:
        x_batch = (x_batch - mean[None, None, None, :]) / std[None, None, None, :]

      yield jnp.array(x_batch, dtype=jnp.float32), jnp.array(y_batch, dtype=jnp.int32)

  return generator(), num_classes, ds_size


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
  return accuracy

def compute_accuracy(state, dataloader):
  """
  Computes accuracy for the given model parameters and dataloader.

  Args:
      params: Model parameters.
      model: Flax model.
      dataloader: DataLoader for the dataset.

  Returns:
      Accuracy as a percentage (float).
  """
  correct_predictions = 0
  total_samples = 0

  for x_batch, y_batch in dataloader:
    # Compute model predictions
    batch_size = y_batch.shape[0]
    correct_predictions += compute_batch_accuracy(state, x_batch, y_batch) * batch_size
    total_samples += batch_size
    if total_samples >= 10000:
      break

  # Compute accuracy as a percentage
  accuracy = (correct_predictions / total_samples)
  return accuracy


 # Prepare the train state for the model parameters
  # (Using Flax's train_state for convenience)
class TrainState(train_state.TrainState):
  apply_inf_fn: Callable
  batch_stats: Any

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
  opt_state_dir = os.path.join(trainstate_dir, "opt_state")
  batch_stats_dir = os.path.join(trainstate_dir, "batch_stats")

  precond_dir = os.path.join(parent_dir, "preconditioner")
  os.makedirs(trainstate_dir, exist_ok=True)
  os.makedirs(params_dirs, exist_ok=True)
  os.makedirs(opt_state_dir, exist_ok=True)
  os.makedirs(batch_stats_dir, exist_ok=True)
  os.makedirs(precond_dir, exist_ok=True)

  # Use the existing save function
  save_pytree_state(params_dirs, trainstate.params)
  save_pytree_state(opt_state_dir, trainstate.opt_state)
  save_pytree_state(batch_stats_dir, trainstate.batch_stats)
  save_pytree_state(precond_dir, preconditioner_blocks)


def restore_trainstate_and_precond(parent_dir: str):
  # Directories for params, state, and opt_state
  trainstate_dir = os.path.join(parent_dir, "trainstate")
  precond_dir = os.path.join(parent_dir, "preconditioner")
  params_dirs = os.path.join(trainstate_dir, "params")
  opt_state_dir = os.path.join(trainstate_dir, "opt_state")
  batch_stats_dir = os.path.join(trainstate_dir, "batch_stats")

  # Use the existing restore function
  restored_params = restore_pytree_state(params_dirs)
  restored_opt_state = restore_pytree_state(opt_state_dir)
  restored_batch_stats = restore_pytree_state(batch_stats_dir)
  restored_preconditioner_blocks = restore_pytree_state(precond_dir)

  return {"params": restored_params, "opt_state": restored_opt_state, "batch_stats":restored_batch_stats}, restored_preconditioner_blocks

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
