""" Various utilities functions for LQR optimization"""
import time
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

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


def prepare_dataloader(batch_size=128, train=True, dataset='mnist'):
  """
  Creates a generator that yields (x, y) from the specified dataset (MNIST, CIFAR-10, or CIFAR-100).
  Returns the generator along with the number of classes in the dataset.
  """
  if dataset == 'mnist':
    ds_name = 'mnist'
    mean = 0.1307  # MNIST mean
    std = 0.3081   # MNIST std
    num_classes = 10
  elif dataset == 'cifar-10':
    ds_name = 'cifar10'
    mean = jnp.array([0.4914, 0.4822, 0.4465])  # CIFAR-10 mean per channel
    std = jnp.array([0.2470, 0.2435, 0.2616])   # CIFAR-10 std per channel
    num_classes = 10
  elif dataset == 'cifar-100':
    ds_name = 'cifar100'
    mean = jnp.array([0.5071, 0.4867, 0.4408])  # CIFAR-100 mean per channel
    std = jnp.array([0.2675, 0.2565, 0.2761])   # CIFAR-100 std per channel
    num_classes = 100
  else:
    raise ValueError("Unsupported dataset. Choose either 'mnist', 'cifar-10', or 'cifar-100'")

  ds, info = tfds.load(ds_name, split='train' if train else 'test', as_supervised=True, with_info=True)

  # Shuffle, batch, repeat
  ds = ds.shuffle(10_000).batch(batch_size).repeat()
  ds = ds.prefetch(tf.data.AUTOTUNE)

  def generator():
    for x_batch, y_batch in ds:
      x_batch = x_batch.numpy()
      y_batch = y_batch.numpy()

      if dataset == 'mnist':
        x_batch = x_batch.astype(jnp.float32) / 255.0  # Normalize to [0,1]
        x_batch = (x_batch - mean) / std  # Standardize
      else:  # CIFAR-10 or CIFAR-100
        x_batch = x_batch.astype(jnp.float32) / 255.0  # Normalize to [0,1]
        x_batch = (x_batch - mean[None, None, None, :]) / std[None, None, None, :]  # Standardize per channel

      yield jnp.array(x_batch, dtype=jnp.float32), jnp.array(y_batch, dtype=jnp.int32)

  return generator(), num_classes


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
