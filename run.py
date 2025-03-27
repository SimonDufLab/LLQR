"""Python script to run experiments"""
import os
import time
from datetime import timedelta
import signal

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
# os.environ["XLA_FLAGS"] = "--xla_dump_hlo_as_text --xla_force_host_platform_device_count=1"  # Logging XLA compilation, for debugging

import jax
import jax.numpy as jnp

import hydra
from aim import Run
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from lqr_optimizer._src.configs.config import model_choice, divergence_choice
from lqr_optimizer._src.preconditioner import BasePreconditioner
from lqr_optimizer._src.utils.utils import cross_entropy_loss, prepare_dataloader, loss_eval, compute_accuracy, compute_batch_accuracy
import lqr_optimizer._src.utils.utils as utl


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
  # Print the loaded config
  print(OmegaConf.to_yaml(cfg))
  experiment_name = cfg.dataset.name + "_" + cfg.architecture.name
  print("Experiment name: {}".format(experiment_name))

  # Check for checkpoints
  load_from_preexisting_model_state = False
  if cfg.preempt_handling:
    SCRATCH = Path(os.environ["SCRATCH"])
    if cfg.jobid:
      SLURM_JOBID = str(cfg.jobid) # ensure string format, avoid int
    else:
      SLURM_JOBID = os.environ["SLURM_JOBID"]
      cfg.jobid = SLURM_JOBID
    saving_dir = SCRATCH / experiment_name / SLURM_JOBID

    # Create the directory if it does not exist
    os.makedirs(saving_dir, exist_ok=True)

    # Check for previous checkpoints
    run_state = utl.load_run_state(saving_dir)
    if run_state:
      load_from_preexisting_model_state = True
    else:  # Initialize the run_state
      run_state = utl.RunState(epoch=0, training_step=0, model_dir=saving_dir,
                               aim_hash=None, slurm_jobid=SLURM_JOBID, exp_name=experiment_name,
                               dropout_key=jax.random.PRNGKey(cfg.rng_seed),
                               best_accuracy=0.0, training_time=0.0)

    aim_hash = run_state["aim_hash"]
  else:
    aim_hash = None

  # Initialize aim for logging
  run = Run(experiment=experiment_name, run_hash=aim_hash, force_resume=True)
  run["config"] = OmegaConf.to_container(cfg)
  if cfg.preempt_handling:
    run_state["aim_hash"] = run.hash

  # 1) Create the data generator
  dataloader, num_classes, train_ds_size = prepare_dataloader(batch_size=cfg.batch_size, train=True, dataset=cfg.dataset.name, augment_dataset=cfg.dataset.augment_dataset)
  test_dataloader, _, _ = prepare_dataloader(batch_size=cfg.batch_size, train=False, dataset=cfg.dataset.name)
  steps_per_epoch = train_ds_size // cfg.batch_size

  # 2) Define model
  model, inf_model = model_choice[cfg.architecture.name](num_classes=num_classes)
  if inf_model is None:
    inf_model = model

  # 3) Initialize model parameters
  rng = jax.random.PRNGKey(cfg.init_key)
  variables = model.init(rng, next(dataloader)[0])
  params = variables['params']
  init_batch_stats = variables.get('batch_stats', {})
  print(jax.tree_util.tree_map(jnp.shape, params))
  print(jax.tree_util.tree_map(jnp.shape, init_batch_stats))

  # 4) Create the main optimizer
  model_optimizer = utl.load_main_optimizer(cfg)

  if load_from_preexisting_model_state:
    state, precond_blocks = utl.restore_trainstate_and_precond(run_state["model_dir"])
    state = utl.TrainState.create(
      apply_fn=model.apply,
      apply_inf_fn=inf_model.apply,
      params=state["params"],
      tx=model_optimizer,
      opt_state=state["opt_state"],
      batch_stats=state["batch_stats"],
    )
  else:
    state = utl.TrainState.create(
      apply_fn=model.apply,
      apply_inf_fn=inf_model.apply,
      params=params,
      tx=model_optimizer,
      batch_stats=init_batch_stats
    )

  # 5) Create the BasePreconditioner
  precond_optimizer = utl.load_precond_optimizer(cfg)

  # Initialize BasePreconditioner
  divergence_f = divergence_choice[cfg.divergence]
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
  if load_from_preexisting_model_state:
    preconditioner.load_blocks(precond_blocks)
    del precond_blocks

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

  signal.signal(signal.SIGTERM, utl.signal_handler)  # Before getting pre-empted and requeued.
  signal.signal(signal.SIGUSR1, utl.signal_handler)  # Before reaching the end of the time limit.

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

  dropout_key = jax.random.PRNGKey(cfg.rng_seed)
  if load_from_preexisting_model_state:
    dropout_key = run_state["dropout_key"]
    starting_step = run_state["training_step"]
    best_acc = run_state["best_accuracy"]
    prev_elapsed_time = run_state["training_time"]
    load_from_preexisting_model_state = False
  else:
    starting_step = 0
    best_acc = 0
    prev_elapsed_time = 0

  print(f"Continuing training from step {starting_step}")
  for step in range(starting_step, cfg.total_steps):
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
      elapsed_time = time.time() - start_time + prev_elapsed_time # Calculate elapsed time
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
      elapsed_time = time.time() - start_time + prev_elapsed_time
      run.track(test_accuracy, name="test accuracy", step=step)
      run.track(test_loss, name="test loss", step=step)
      print("============================")
      print(f"Step {step} | Test Loss: {test_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
      print(f"Step {step} | Test Accuracy: {test_accuracy:.2f}%")
      print(f"Test accuracy across entire dataset computed in {time.time() - test_time_start:.2f} seconds")
      print("============================")
    if (step > 0) and cfg.preempt_handling and (step % cfg.checkpoint_freq == 0):
      chckpt_init_time = time.time()
      elapsed_time = time.time() - start_time + prev_elapsed_time
      utl.checkpoint_exp(run_state, state, preconditioner.expose_blocks(), curr_epoch=step // steps_per_epoch,
                         curr_step=step, dropout_key=dropout_key, best_acc=best_acc,
                         training_time=elapsed_time)
      print(
        f"Checkpointing performed in: {timedelta(seconds=time.time() - chckpt_init_time)}")


  print("Training complete!")
  # End timer
  total_time = time.time() - start_time + prev_elapsed_time
  print(f"Training complete! Total Time Elapsed: {total_time:.2f} seconds")


if __name__ == "__main__":
  main()
