"""Python script to run experiments"""
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import jax
import jax.numpy as jnp
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training import train_state
from aim import Run

from lqr_optimizer._src.models.mlp import create_mlp
from lqr_optimizer._src.preconditioner import BasePreconditioner


# Divergence function (given in the prompt):
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

def loss_to_params(params, apply_fn, x, y):
  """
  params: parameters of the model
  apply_fn: function to apply model (Flax typically provides model.apply)
  x: input data
  y: integer class labels
  """
  log_probs = apply_fn({'params': params}, x)  # shape [batch_size, num_classes]
  # We expect the final layer to produce log-softmax output (as defined in MLP),

  return cross_entropy_loss(log_probs, y)

# A small utility to batch the MNIST dataset
def prepare_dataloader(batch_size=128, train=True):
  """
  Creates a generator that yields (x, y) from the MNIST dataset.
  Each iteration yields a single batch (numpy arrays).
  """
  ds, info = tfds.load('mnist', split='train' if train else 'test', as_supervised=True, with_info=True)

  # Shuffle, batch, repeat
  ds = ds.shuffle(10_000).batch(batch_size).repeat()
  ds = ds.prefetch(tf.data.AUTOTUNE)

  for x_batch, y_batch in ds:
    # Convert TF Tensors -> NumPy arrays (JAX can handle them, but let's be explicit)
    yield (jnp.array(x_batch.numpy().reshape(-1, 28 * 28)/255, dtype=jnp.float32),
           jnp.array(y_batch.numpy(), dtype=jnp.int32))

def compute_batch_accuracy(params, apply_fn, x_batch, y_batch):
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
  log_probs = apply_fn({'params': params}, x_batch)  # shape [batch_size, num_classes]

  # Predicted class (argmax of logits)
  predictions = jnp.argmax(jnp.exp(log_probs), axis=1)

  # Compare predictions with ground truth
  correct_predictions = jnp.sum(predictions == y_batch)

  # Compute accuracy
  accuracy = (correct_predictions / y_batch.shape[0]) * 100
  return accuracy

def compute_accuracy(params, apply_fn, dataloader):
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
    correct_predictions += compute_batch_accuracy(params, apply_fn, x_batch, y_batch) * batch_size
    total_samples += batch_size
    if total_samples >= 10000:
      break

  # Compute accuracy as a percentage
  accuracy = (correct_predictions / total_samples)
  return accuracy


def main():
  # Initialize aim for logging
  run = Run()
  # Hyperparameters
  batch_size = 128
  learning_rate = 1e-3
  momentum = 0.9
  t = 5000  # total training iterations
  update_preconditioner_every = 500  # k: update the preconditioner every k steps
  precond_steps = 25  # how many gradient steps to take on the preconditioner
  precond_lr = 1e-1  # learning rate for the preconditioner's ADAM
  test_eval_freq = 500
  use_preconditioner = True

  run["hparams"] = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "momentum": momentum,
    "update_preconditioner_every": update_preconditioner_every,
    "precond_steps": precond_steps,
    "precond_lr": precond_lr,
    "total_steps": t,
    "test_eval_freq": test_eval_freq,
    "use_preconditioner": use_preconditioner,
  }

  # 1) Create MNIST data generator
  dataloader = prepare_dataloader(batch_size=batch_size, train=True)
  test_dataloader = prepare_dataloader(batch_size=batch_size, train=False)

  # 2) Define model
  num_classes = 10
  model = create_mlp(num_classes=num_classes)

  # 3) Initialize model parameters
  rng = jax.random.PRNGKey(42)
  dummy_x = jnp.ones((1, 28 * 28))  # dummy input for shape inference
  params = model.init(rng, dummy_x)#['params']
  print(jax.tree_map(jnp.shape, params))

  # 4) Create the main optimizer (SGD with momentum)
  # model_optimizer = optax.sgd(learning_rate=learning_rate, momentum=momentum)
  model_optimizer = optax.adam(learning_rate=learning_rate)

  # Prepare the train state for the model parameters
  # (Using Flax's train_state for convenience)
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=model_optimizer
  )

  # 5) Create the BasePreconditioner
  block_structure = 'diagonal'
  block_structure_init = 'identity'
  precond_solver = "adam"
  if precond_solver == "adam":
    optax_solver_for_precond = optax.adam(precond_lr)
  elif precond_solver == "momentum":
    optax_solver_for_precond = optax.sgd(precond_lr, momentum=momentum)
  elif precond_solver == "sgd":
    optax_solver_for_precond = optax.sgd(precond_lr)
  multibatch_training = False # Use multiple batches for a single preconditioner update or not

  run["hparams"].update({"block_structure": block_structure,
                         "block_structure_init": block_structure_init,
                         "multibatch_training": multibatch_training})

  # Initialize BasePreconditioner
  preconditioner = BasePreconditioner(
    divergence_function=divergence_f,
    loss_fn=cross_entropy_loss,
    block_structure=block_structure,
    block_structure_init=block_structure_init,
    model=model,
    network_params=params,
    optax_solver=optax_solver_for_precond,
    damping=0.0,
    divergence_args_index=-1
  )

  # ---------------------------------------------------------------------------------
  # Training loop
  # ---------------------------------------------------------------------------------

  @jax.jit
  def compute_grad(params, x, y):
    """Compute standard gradient of the cross entropy loss."""
    return jax.grad(loss_to_params, argnums=0)(params, model.apply, x, y)

  # Start timer
  start_time = time.time()
  # We'll keep a local dataloader iterator
  data_iter = dataloader  # generator

  for step in range(t):
    # Possibly update the preconditioner every `update_preconditioner_every` steps
    if (step % update_preconditioner_every) == 0 and use_preconditioner:
      # The preconditioner update can be run on a mini-batch from the dataloader
      # We do multiple steps (precond_steps) of "preconditioner training"
      preconditioner.update_preconditioner(state.params, precond_steps, data_iter, multibatch_training)

    # Grab the next batch for normal training
    x_batch, y_batch = next(data_iter)

    # 1) Compute the raw gradient
    grads = compute_grad(state.params, x_batch, y_batch)

    # 2) Apply the preconditioner on the gradient
    precond_grads = preconditioner.apply(grads)

    # 3) Use the preconditioned gradient to update the model with normal SGD
    state = state.apply_gradients(grads=precond_grads)

    # Logging or testing every so often
    if step % 10 == 0:
      # Simple logging
      train_loss = loss_to_params(state.params, model.apply, x_batch, y_batch)
      elapsed_time = time.time() - start_time  # Calculate elapsed time
      run.track(train_loss, name="train loss", step=step)
      # Compute batch accuracy
      batch_accuracy = compute_batch_accuracy(state.params, model.apply, x_batch, y_batch)
      run.track(batch_accuracy, name="train accuracy", step=step)
      if step % 200 == 0:
        # Print info
        print(f"Step {step} | Train Loss: {train_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Step {step} | Batch Accuracy: {batch_accuracy:.2f}%")
    if step % test_eval_freq == 0:
      test_accuracy = compute_accuracy(state.params, model.apply, test_dataloader)
      run.track(test_accuracy, name="test accuracy", step=step)


  print("Training complete!")
  # End timer
  total_time = time.time() - start_time
  print(f"Training complete! Total Time Elapsed: {total_time:.2f} seconds")


if __name__ == "__main__":
  main()
