"""Python script to run experiments on single layer function to test the validity of the lqr approximation"""
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from aim import Run
from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree

from lqr_optimizer._src.models.single_layer_functions import *
from lqr_optimizer._src.preconditioner import BasePreconditioner
from lqr_optimizer._src.exact_methods import make_newton_step

def loss_fn(y, target):
  return jnp.sum(y)

def loss_to_params(params, apply_fn, x, y):
  return loss_fn(apply_fn({'params': params}, x), y)

model_dict = {"rosenbrock": get_rosenbrock_model_and_datagen,
              "split_rosenbrock": get_split_rosenbrock_model_and_datagen,
              "ackley": get_ackley_model_and_datagen,
              "goldstein_price": get_goldsteinprice_model_and_datagen}

def main():
  # Initialize aim for logging
  problem_type = "split_rosenbrock"

  log_path = "./single-layer-test"
  run = Run(repo=log_path, experiment=problem_type)
  # Hyperparameters
  batch_size = 1
  learning_rate = 1e-3
  momentum = 0.9
  optimizer = "sgd"
  t = 5000  # total training iterations
  update_preconditioner_every = 1  # k: update the preconditioner every k steps
  precond_steps = 500 # how many gradient steps to take on the preconditioner
  precond_lr = 1e-3  # learning rate for the preconditioner's ADAM
  test_eval_freq = 5
  damping = 0.0
  exact_newton = False
  use_preconditioner = True
  precond_clip_norm = 1e-3
  normalize_grad_for_lqr = False

  if exact_newton: # Don't use preconditioner when exact solving
    use_preconditioner = False

  optimizer_dict = {"sgd": optax.sgd,
                    "momentum": Partial(optax.sgd, momentum=momentum),
                    "adam": optax.adam, }

  # Create model
  model, dataloader = model_dict[problem_type]()

  # Initialize model parameters
  rng = jax.random.PRNGKey(42)
  dummy_x = jnp.ones((1, 28 * 28))  # dummy input for shape inference
  params = model.init(rng, dummy_x)['params']
  print(jax.tree_map(jnp.shape, params))

  # Create the optimizer
  model_optimizer = optimizer_dict[optimizer](learning_rate)

  # Prepare the train state for the model parameters
  # (Using Flax's train_state for convenience)
  state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=model_optimizer
  )
  print(type(state.params))

  # 5) Create the BasePreconditioner
  block_structure = 'dense'
  block_structure_init = 'identity'
  precond_solver = "momentum"
  if precond_solver == "adam":
    optax_solver_for_precond = optax.adam(precond_lr)
  elif precond_solver == "momentum":
    optax_solver_for_precond = optax.sgd(precond_lr, momentum=momentum)
  elif precond_solver == "sgd":
    optax_solver_for_precond = optax.sgd(precond_lr)
  multibatch_training = False  # Use multiple batches for a single preconditioner update or not

  run["hparams"] = {
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "momentum": momentum,
    "update_preconditioner_every": update_preconditioner_every,
    "precond_steps": precond_steps,
    "precond_lr": precond_lr,
    "precond_clip_norm": precond_clip_norm,
    "total_steps": t,
    "test_eval_freq": test_eval_freq,
    "damping": damping,
    "exact_newton": exact_newton,
    "use_preconditioner": use_preconditioner,
    "precond_solver": precond_solver,
    "block_structure": block_structure,
    "block_structure_init": block_structure_init,
    "multibatch_training": multibatch_training,
    "normalize_grad_for_lqr": normalize_grad_for_lqr,
  }

  # Initialize BasePreconditioner
  preconditioner = BasePreconditioner(
    divergence_function=None,
    loss_fn=loss_fn,
    block_structure=block_structure,
    block_structure_init=block_structure_init,
    model=model,
    network_params=params,
    optax_solver=optax_solver_for_precond,
    trainstate_solver=state.tx,
    damping=damping,
    divergence_args_index=None,
    multibatch=multibatch_training,
    precond_clip_norm = precond_clip_norm,
    normalize_grad_for_lqr= normalize_grad_for_lqr,
    preconditioner_update_steps = precond_steps,
  )

  # ---------------------------------------------------------------------------------
  # Training loop
  # ---------------------------------------------------------------------------------

  @jax.jit
  def compute_grad(_params, x, y):
    """Compute standard gradient of the cross entropy loss."""
    return jax.grad(loss_to_params, argnums=0)(_params, model.apply, x, y)

  # For exact Newton
  newton_step = make_newton_step(
    loss_to_params, model.apply, damping=0.0, tol=1e-5,
  )

  # Start timer
  start_time = time.time()
  # We'll keep a local dataloader iterator
  data_iter = dataloader  # generator

  for step in range(t):
    # Possibly update the preconditioner every `update_preconditioner_every` steps
    if (step % update_preconditioner_every) == 0 and use_preconditioner:
      # The preconditioner update can be run on a mini-batch from the dataloader
      # We do multiple steps (precond_steps) of "preconditioner training"
      preconditioner.update_preconditioner(state.params, data_iter, state.opt_state)

    # Grab the next batch for normal training
    x_batch, y_batch = next(data_iter)

    if exact_newton:
      precond_grads, _ = newton_step(state.params, x_batch, y_batch)

    else:
      # 1) Compute the raw gradient
      grads = compute_grad(state.params, x_batch, y_batch)

      # 2) Apply the preconditioner on the gradient
      precond_grads = preconditioner.apply(grads)

    # 3) Use the preconditioned gradient to update the model with normal SGD
    state = state.apply_gradients(grads=precond_grads)

    # Logging or testing every so often
    if step % test_eval_freq == 0:
      # Simple logging
      train_loss = loss_to_params(state.params, model.apply, x_batch, y_batch)
      elapsed_time = time.time() - start_time  # Calculate elapsed time
      run.track(train_loss, name="train loss", step=step)
      if step % 200 == 0:
        # Print info
        print(f"Step {step} | Train Loss: {train_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"‖update‖ {jnp.linalg.norm(ravel_pytree(precond_grads)[0]):.4f}")
        if use_preconditioner:
          print(f"Preconditioners:")
          print(preconditioner.expose_blocks())
          print()

  print("Training complete!")
  # End timer
  total_time = time.time() - start_time
  print(f"Training complete! Total Time Elapsed: {total_time:.2f} seconds")

if __name__ == "__main__":
  main()