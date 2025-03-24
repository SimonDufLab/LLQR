"""Python script to run experiments"""
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
# os.environ["XLA_FLAGS"] = "--xla_dump_hlo_as_text --xla_force_host_platform_device_count=1"  # Logging XLA compilation, for debugging

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state
from typing import Any, Callable

import hydra
from aim import Run
from omegaconf import DictConfig, OmegaConf

from lqr_optimizer._src.configs.config import model_choice
from lqr_optimizer._src.preconditioner import BasePreconditioner
from lqr_optimizer._src.utils.utils import load_main_optimizer, load_precond_optimizer


# Divergence function (for NGD):
def divergence_f(px, px_):
  # Taking into account we return log-softmax
  return (-px * jnp.log(px_)).sum()


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


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
  # Print the loaded config
  print(OmegaConf.to_yaml(cfg))

  # Initialize aim for logging
  run = Run()
  run["config"] = OmegaConf.to_container(cfg)

  # 1) Create the data generator
  dataloader, num_classes = prepare_dataloader(batch_size=cfg.batch_size, train=True, dataset=cfg.dataset)
  test_dataloader, _ = prepare_dataloader(batch_size=cfg.batch_size, train=False, dataset=cfg.dataset)

  # 2) Define model
  model, inf_model = model_choice[cfg.architecture](num_classes=num_classes)
  if inf_model is None:
    inf_model = model

  # 3) Initialize model parameters
  rng = jax.random.PRNGKey(cfg.init_key)
  variables = model.init(rng, next(dataloader)[0])
  params = variables['params']
  init_batch_stats = variables.get('batch_stats', {})
  print(jax.tree_map(jnp.shape, params))
  print(jax.tree_map(jnp.shape, init_batch_stats))

  # 4) Create the main optimizer
  model_optimizer = load_main_optimizer(cfg)

  # Prepare the train state for the model parameters
  # (Using Flax's train_state for convenience)
  class TrainState(train_state.TrainState):
    apply_inf_fn: Callable
    batch_stats: Any

  state = TrainState.create(
    apply_fn=model.apply,
    apply_inf_fn=inf_model.apply,
    params=params,
    tx=model_optimizer,
    batch_stats = init_batch_stats
  )

  # 5) Create the BasePreconditioner

  precond_optimizer = load_precond_optimizer(cfg)

  # Initialize BasePreconditioner
  preconditioner = BasePreconditioner(
    divergence_function=divergence_f,
    loss_fn=cross_entropy_loss,
    block_structure=cfg.block_structure,
    block_structure_init=cfg.block_structure_init,
    model=inf_model,
    network_params=params,
    optax_solver=precond_optimizer,
    precond_clip_norm=cfg.precond_clip_norm,
    preconditioner_update_steps=cfg.precond_steps,
    multibatch=cfg.multibatch_training,
    damping=cfg.damping,
    divergence_args_index=-1
  )

  # ---------------------------------------------------------------------------------
  # Training loop
  # ---------------------------------------------------------------------------------

  # @jax.jit
  def loss_fn(params, apply_fn, _batch_stats, x, y):
    # Pass both params and batch_stats, and mark batch_stats as mutable.
    (log_probs, new_model_state) = apply_fn(
      {'params': params, 'batch_stats': _batch_stats},
      x,
      mutable=['batch_stats']
    )
    loss = cross_entropy_loss(log_probs, y)
    return loss, new_model_state

  @jax.jit
  def compute_updates(params, _batch_stats, x, y):
    """Compute standard gradient of the cross entropy loss."""
    return jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params, model.apply, _batch_stats, x, y)

  # @jax.jit
  def train_step(state, x, y):
    # @jax.jit
    # def loss_fn(params):
    #   # Pass both params and batch_stats, and mark batch_stats as mutable.
    #   (log_probs, new_model_state) = state.apply_fn(
    #     {'params': params, 'batch_stats': state.batch_stats},
    #     x,
    #     mutable=['batch_stats']
    #   )
    #   loss = cross_entropy_loss(log_probs, y)
    #   return loss, new_model_state

    # Compute the loss and gradients. Note: we use has_aux=True to get new_model_state.
    # (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    (loss, new_model_state), grads = compute_updates(state.params, state.batch_stats, x, y)

    # 2) Apply the preconditioner on the gradient
    precond_grads = preconditioner.apply(grads)

    new_state = state.apply_gradients(grads=precond_grads, batch_stats=new_model_state['batch_stats'])
    return new_state, loss

  # Start timer
  start_time = time.time()
  # We'll keep a local dataloader iterator
  data_iter = dataloader  # generator

  for step in range(cfg.total_steps):
    # Possibly update the preconditioner every `update_preconditioner_every` steps
    if (step % cfg.update_preconditioner_every) == 0 and cfg.use_preconditioner:
      # The preconditioner update can be run on a mini-batch from the dataloader
      # We do multiple steps (precond_steps) of "preconditioner training"
      precond_update_start_time = time.time()
      preconditioner.update_preconditioner(state.params, data_iter,
                                           other_model_variables={'batch_stats': state.batch_stats})
      precond_max, precond_min, precond_norm = preconditioner.get_stats()
      # !!Remove below when timing against non-2nd order methods!! (Affect computation time)
      run.track(precond_max, name="Maximum across preconditioner", step=step)
      run.track(precond_min, name="Minimum across preconditioner", step=step)
      run.track(precond_norm, name="Preconditioner l2 norm", step=step)
      print(f"Preconditioner was updated in {time.time()-precond_update_start_time:.2f} seconds")

    # Grab the next batch for normal training
    x_batch, y_batch = next(data_iter)

    state, loss = train_step(state, x_batch, y_batch)

    # Logging or testing every so often
    if step % 10 == 0:
      # Simple logging
      # train_loss = loss_eval(state, x_batch, y_batch)
      train_loss = loss
      elapsed_time = time.time() - start_time  # Calculate elapsed time
      run.track(train_loss, name="train loss", step=step)
      # Compute batch accuracy
      batch_accuracy = compute_batch_accuracy(state, x_batch, y_batch)
      run.track(batch_accuracy, name="train accuracy", step=step)
      if step % 200 == 0:
        # Print info
        print(f"Step {step} | Train Loss: {train_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Step {step} | Batch Accuracy: {batch_accuracy:.2f}%")
    if step % cfg.test_eval_freq == 0:
      test_time_start = time.time()
      test_accuracy = compute_accuracy(state, test_dataloader)
      x_test, y_test = next(test_dataloader)
      test_loss = loss_eval(state, x_test, y_test)
      elapsed_time = time.time() - start_time
      run.track(test_accuracy, name="test accuracy", step=step)
      run.track(test_loss, name="test loss", step=step)
      print("============================")
      print(f"Step {step} | Test Loss: {test_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
      print(f"Step {step} | Test Accuracy: {test_accuracy:.2f}%")
      print(f"Test accuracy across entire dataset computed in {time.time() - test_time_start:.2f} seconds")
      print("============================")


  print("Training complete!")
  # End timer
  total_time = time.time() - start_time
  print(f"Training complete! Total Time Elapsed: {total_time:.2f} seconds")


if __name__ == "__main__":
  main()
