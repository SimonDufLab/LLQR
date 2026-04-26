"""Python script to run experiments"""
import inspect
import math
import os
import time
from datetime import timedelta
import signal
import optax
import tensorflow as tf

tf.config.experimental.set_visible_devices([], "GPU")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
# os.environ["XLA_FLAGS"] = "--xla_dump_hlo_as_text --xla_force_host_platform_device_count=1"  # Logging XLA compilation, for debugging

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from jax.flatten_util import ravel_pytree

import hydra
from aim import Run
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from lqr_optimizer._src.configs.config import model_choice, divergence_choice, lr_schedule_choice
from lqr_optimizer._src.preconditioner import BasePreconditioner
from lqr_optimizer._src.utils.utils import cross_entropy_loss, prepare_dataloader, compute_accuracy_and_loss, compute_batch_accuracy, compute_accuracy_and_loss_with_hists
from lqr_optimizer._src.utils.dataloaders.hf_loaders import prepare_hf_dataset
from lqr_optimizer._src.utils.dataloaders.iwslt14_de_en import (
  load_iwslt14_de_en_dictionaries,
  prepare_local_seq2seq_dataset,
)
from lqr_optimizer._src.utils.seq2seq_utils import (
  SEQ2SEQ_TASK_KIND,
  evaluate_translation_generation,
  translation_bleu_eval_mode,
)
import lqr_optimizer._src.utils.sam_mode_handlers as sam_mode_handlers
import lqr_optimizer._src.utils.utils as utl


def should_run_periodic_event(*, step: int, total_steps: int, every: int) -> bool:
  """Skip expensive step-0 periodic work on multi-step runs.

  Single-step runs such as `total_epochs=0` still keep step-0 periodic
  eligibility because there is no later step to observe. The training loop
  separately skips mutating work on its final sentinel.
  """
  return (step % every == 0) and ((step != 0) or (total_steps == 1))


def should_run_training_step(*, step: int, total_steps: int) -> bool:
  """Return whether this loop step should mutate model or preconditioner state."""
  return int(step) < int(total_steps) - 1


def track_translation_token_validation_metrics(
    run,
    *,
    valid_accuracy,
    valid_loss,
    step: int,
    elapsed_time: float,
):
  """Track translation token validation metrics, including fairseq NLL bits/token."""
  valid_nll_loss_bits = float(valid_loss) / math.log(2.0)
  time_step = elapsed_time * 100
  run.track(valid_accuracy, name="valid token_accuracy", step=step)
  run.track(valid_loss, name="valid loss", step=step)
  run.track(valid_nll_loss_bits, name="valid_nll_loss", step=step)
  run.track(valid_accuracy, name="valid token_accuracy|t", step=time_step)
  run.track(valid_loss, name="valid loss|t", step=time_step)
  run.track(valid_nll_loss_bits, name="valid_nll_loss|t", step=time_step)


def build_model_pair_from_dataset_info(*, architecture_name: str, num_classes: int, ds_info, architecture_cfg=None):
  """Build `(train_model, inf_model, model_kwargs)` from dataset metadata."""
  model_kwargs = utl.resolve_model_init_kwargs(architecture_name, ds_info)
  if architecture_cfg is not None:
    raw_architecture_cfg = (
      OmegaConf.to_container(architecture_cfg, resolve=True)
      if isinstance(architecture_cfg, DictConfig)
      else dict(architecture_cfg)
    )
    factory_signature = inspect.signature(model_choice[architecture_name])
    supported_keys = set(factory_signature.parameters)
    supported_keys.discard("num_classes")
    accepts_var_keyword = any(
      parameter.kind == inspect.Parameter.VAR_KEYWORD
      for parameter in factory_signature.parameters.values()
    )
    for key, value in raw_architecture_cfg.items():
      if key == "name" or value is None:
        continue
      if accepts_var_keyword or key in supported_keys:
        model_kwargs[key] = value
  model, inf_model = model_choice[architecture_name](num_classes=num_classes, **model_kwargs)
  if inf_model is None:
    inf_model = model
  return model, inf_model, model_kwargs


def build_translation_next_log_probs_fn(*, model, state):
  @jax.jit
  def apply_next_log_probs(variables, src_tokens, prev_output_tokens):
    return model.apply(
      variables,
      (src_tokens, prev_output_tokens),
      method=model.next_log_probs,
      mutable=False,
    )

  def next_log_probs(src_tokens, prev_output_tokens):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    return apply_next_log_probs(variables, src_tokens, prev_output_tokens)

  return next_log_probs


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
  assert 0.0 <= cfg.ema_decay <= 1.0, f"cfg.ema_decay ({cfg.ema_decay}) must be in [0, 1]"
  # Print the loaded config
  print(OmegaConf.to_yaml(cfg))
  experiment_name = cfg.dataset.name + "_" + cfg.architecture.name
  print("Experiment name: {}".format(experiment_name))
  backend_name = jax.default_backend()
  backend_devices = [str(device) for device in jax.devices()]
  print(f"JAX backend: {backend_name}")
  print(f"JAX devices: {backend_devices}")

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
                               best_metric_name="accuracy", best_metric_value=0.0,
                               best_metric_step=-1, best_accuracy=0.0, training_time=0.0)

    aim_hash = run_state["aim_hash"]
  else:
    aim_hash = None

  # 1a) Create the data generators
  train_eval_dataloader = None
  if not cfg.eval_batch_size:
    cfg.eval_batch_size = cfg.batch_size
  if cfg.dataset.loader == "tfds":
    dataloader, ds_info = prepare_dataloader(
      batch_size=cfg.batch_size,
      train=True,
      dataset=cfg.dataset.name,
      augment_dataset=cfg.dataset.augment_dataset,
      lt_config=cfg.dataset.lt_config,
      dataset_dir=cfg.dataset.dataset_dir,
      batch_overlap_fraction=cfg.batch_overlap_fraction,  # NEW
    )
    if cfg.add_train_eval:
      train_eval_dataloader, _ = prepare_dataloader(
        batch_size=cfg.eval_batch_size,
        train=True,
        dataset=cfg.dataset.name,
        augment_dataset=False,
        lt_config=cfg.dataset.lt_config,  # IMPORTANT: same LT subset if you train LT
        dataset_dir=cfg.dataset.dataset_dir,
        batch_overlap_fraction=0.0,
        shuffle=False,
      )
    # precond_dataloader, _ = prepare_dataloader(batch_size=cfg.precond_batch_size, train=True, dataset=cfg.dataset.name, augment_dataset=cfg.dataset.augment_dataset, lt_config=cfg.dataset.lt_config, dataset_dir=cfg.dataset.dataset_dir)
    test_dataloader, test_ds_info = prepare_dataloader(batch_size=cfg.eval_batch_size, train=False, dataset=cfg.dataset.name, dataset_dir=cfg.dataset.dataset_dir)
    ds_info = dict(ds_info)
    ds_info.setdefault("test_ds_size", test_ds_info["ds_size"])
  elif cfg.dataset.loader == "hf":
    dataloader, test_dataloader, ds_info = prepare_hf_dataset(cfg.dataset.name)(
      save_path = Path(cfg.dataset.dataset_dir),
      tokenizers_path = Path(cfg.dataset.tokenizer_dir),
      batch_size = cfg.batch_size,
      bptt = cfg.dataset.target_len,
      eval_batch_size = cfg.eval_batch_size,
      )
    ds_info = dict(ds_info)
  elif cfg.dataset.loader == "local_seq2seq_text":
    if cfg.add_train_eval:
      raise ValueError("`add_train_eval` is not yet supported for local_seq2seq_text datasets.")
    numeric_cache_dir = cfg.dataset.numeric_cache_dir
    dataloader, test_dataloader, ds_info = prepare_local_seq2seq_dataset(cfg.dataset.name)(
      dataset_dir=Path(cfg.dataset.dataset_dir),
      numeric_cache_dir=Path(numeric_cache_dir) if numeric_cache_dir else None,
      max_tokens=cfg.dataset.max_tokens,
      eval_batch_size=cfg.eval_batch_size,
      prefetch=cfg.dataset.prefetch,
      eval_prefetch=cfg.dataset.eval_prefetch,
      train_shape_bucket_mode=cfg.dataset.train_shape_bucket_mode,
      train_shape_bucket_count=cfg.dataset.train_shape_bucket_count,
      source_lang=cfg.dataset.source_lang,
      target_lang=cfg.dataset.target_lang,
      padding_factor=cfg.dataset.padding_factor,
    )
    ds_info = dict(ds_info)
  else:
    raise ValueError(f"Loader missing or not supported for {cfg.dataset.name}")
  task_kind = ds_info.get("task_kind")
  is_translation_task = task_kind == SEQ2SEQ_TASK_KIND
  num_classes = ds_info['num_classes']
  translation_target_pad_id = (
    int(ds_info["model_init_kwargs"]["pad_id"])
    if is_translation_task
    else None
  )
  accounting = utl.resolve_runner_accounting(
    ds_info,
    batch_size=cfg.batch_size,
    grad_acc_steps=cfg.grad_acc_steps,
    total_epochs=cfg.total_epochs,
  )
  train_ds_size = accounting["train_ds_size"]
  test_ds_size = accounting["test_ds_size"]
  train_eval_target_count = accounting["train_eval_target_count"]
  test_eval_target_count = accounting["test_eval_target_count"]
  steps_per_epoch_rounded = accounting["steps_per_epoch_rounded"]
  steps_per_epoch = accounting["steps_per_epoch"]
  total_steps = accounting["total_steps"]
  if not cfg.update_preconditioner_until:
    cfg.update_preconditioner_until = total_steps + 1

  translation_eval_assets = None
  if is_translation_task and cfg.translation_eval.enabled:
    if cfg.dataset.name != "iwslt14_de_en":
      raise ValueError(
        "Wave 5 translation evaluation is only implemented for dataset=iwslt14_de_en."
      )
    translation_eval_assets = load_iwslt14_de_en_dictionaries(
      dataset_dir=Path(cfg.dataset.dataset_dir),
      numeric_cache_dir=Path(cfg.dataset.numeric_cache_dir) if cfg.dataset.numeric_cache_dir else None,
      source_lang=cfg.dataset.source_lang,
      target_lang=cfg.dataset.target_lang,
      padding_factor=cfg.dataset.padding_factor,
    )

  # 1b) Initialize aim for logging
  run = Run(repo=cfg.logging.aim_repo, experiment=experiment_name, run_hash=aim_hash, force_resume=True)
  run["config"] = OmegaConf.to_container(cfg)
  run["jax_backend"] = backend_name
  run["jax_devices"] = backend_devices
  if ds_info.get("batch_shape_contract") is not None:
    run["batch_shape_contract"] = ds_info["batch_shape_contract"]
  if ds_info.get("preconditioner_shape_contract") is not None:
    run["preconditioner_shape_contract"] = ds_info["preconditioner_shape_contract"]
  if cfg.preempt_handling:
    run_state["aim_hash"] = run.hash

  # 2) Define model
  model, inf_model, model_kwargs = build_model_pair_from_dataset_info(
    architecture_name=cfg.architecture.name,
    num_classes=num_classes,
    ds_info=ds_info,
    architecture_cfg=cfg.architecture,
  )

  # 3) Initialize model parameters
  rng = jax.random.PRNGKey(cfg.init_key)
  init_x, _ = utl.split_supervised_batch(next(dataloader))
  variables = model.init(rng, init_x)
  params = variables['params']
  init_batch_stats = variables.get('batch_stats', {})
  print(jax.tree_util.tree_map(jnp.shape, params))
  print(jax.tree_util.tree_map(jnp.shape, init_batch_stats))

  # 4) Create the main optimizer
  if cfg.wd_mask:
    mask = utl.mask_from_flat_keys(params, cfg.wd_mask)
  else:
    mask = None
  opt_chain = []
  if cfg.weight_decay and "adamw" not in cfg.main_optimizer:
    opt_chain.append(optax.add_decayed_weights(weight_decay=cfg.weight_decay, mask=mask))
  if cfg.lr_scheduler and cfg.lr_scheduler.name != "constant":
    lr_sched_kwargs = {_key:_value for _key, _value in cfg.lr_scheduler.items() if _key!='name'}
    lr_or_sched = lr_schedule_choice[cfg.lr_scheduler.name](base_lr=cfg.learning_rate, total_epochs=cfg.total_epochs ,steps_per_epoch=steps_per_epoch, **lr_sched_kwargs)
  else: lr_or_sched = cfg.learning_rate
  opt_chain.append(utl.load_main_optimizer(cfg, lr_or_sched, mask=mask))
  model_optimizer = optax.chain(*opt_chain)

  migrated_checkpoint_layout = False
  if load_from_preexisting_model_state:
    restored_state, precond_blocks = utl.restore_trainstate_and_precond(run_state["model_dir"])
    restored_params, restored_batch_stats, migrated_checkpoint_layout = model.maybe_migrate_legacy_checkpoint(
      restored_state["params"], restored_state["batch_stats"], params, init_batch_stats
    )
    if migrated_checkpoint_layout:
      print("Migrated a legacy coarse-stage checkpoint to the split-stage model layout.")
      print("Reinitializing main optimizer state, gbar/g_last, and preconditioner blocks for this resume.")
      state = utl.TrainState.create(
        apply_fn=model.apply,
        apply_inf_fn=inf_model.apply,
        params=restored_params,
        gbar=utl.tree_zeros_like(restored_params),
        g_last=utl.tree_zeros_like(restored_params),
        tx=model_optimizer,
        batch_stats=restored_batch_stats,
      )
      precond_blocks = None
    else:
      state = utl.TrainState.create(
        apply_fn=model.apply,
        apply_inf_fn=inf_model.apply,
        params=restored_state["params"],
        gbar=restored_state["gbar"],
        g_last=restored_state["g_last"],
        tx=model_optimizer,
        opt_state=restored_state["opt_state"],
        batch_stats=restored_state["batch_stats"],
      )
  else:
    state = utl.TrainState.create(
      apply_fn=model.apply,
      apply_inf_fn=inf_model.apply,
      params=params,
      gbar=utl.tree_zeros_like(params),
      g_last=utl.tree_zeros_like(params),
      tx=model_optimizer,
      batch_stats=init_batch_stats
    )

  # Lock the legacy SAM-mode contract before any heavy JIT or runtime setup.
  sam_mode_handlers.validate_sam_mode_contract(cfg, state.opt_state)

  # 5) Create the BasePreconditioner
  if cfg.precond_lr_scheduler and cfg.precond_lr_scheduler.name != "constant":
    precond_lr_sched_kwargs = {_key:_value for _key, _value in cfg.precond_lr_scheduler.items() if _key!='name'}
    precond_lr_fn = lr_schedule_choice[cfg.precond_lr_scheduler.name](base_lr=cfg.precond_lr, total_epochs=cfg.total_epochs ,steps_per_epoch=steps_per_epoch, **precond_lr_sched_kwargs)
    precond_lr = 1.0
  else:
    precond_lr_fn = lambda _: 1.0
    precond_lr = cfg.precond_lr
  precond_optimizer = utl.load_precond_optimizer(cfg, precond_lr)

  # Additional option: schedule on ema_decay
  if cfg.ema_scheduler and cfg.ema_scheduler.name != "constant":
    ema_sched_kwargs = {_key:_value for _key, _value in cfg.ema_scheduler.items() if _key!='name'}
    ema_fn = lr_schedule_choice[cfg.ema_scheduler.name](base_lr=cfg.ema_decay, total_epochs=cfg.total_epochs ,steps_per_epoch=steps_per_epoch, **ema_sched_kwargs)
  else:
    ema_fn = lambda _: cfg.ema_decay

  # Optional: schedule on how often the preconditioner is updated:
  if cfg.precond_update_scheduler and cfg.precond_update_scheduler.name != "constant":
    precond_update_sched_kwargs = {_key:_value for _key, _value in cfg.precond_update_scheduler.items() if _key!='name'}
    precond_up_sched = lr_schedule_choice[cfg.precond_update_scheduler.name](base_lr=cfg.update_preconditioner_every, total_epochs=cfg.total_epochs ,steps_per_epoch=steps_per_epoch, **precond_update_sched_kwargs)
  else:
    precond_up_sched = lambda _: cfg.update_preconditioner_every

  # Initialize BasePreconditioner
  if cfg.divergence == "renyi":
    divergence_kwarg = {"order":cfg.divergence_order_param}
  else:
    divergence_kwarg = {}
  if divergence_kwarg:
    divergence_f = Partial(divergence_choice[cfg.divergence], **divergence_kwarg)
  else:
    divergence_f = divergence_choice[cfg.divergence]
  preconditioner = BasePreconditioner(
    divergence_function=divergence_f,
    loss_fn=Partial(
      cross_entropy_loss,
      label_smoothing=cfg.label_smoothing,
      ignore_index=translation_target_pad_id,
    ),
    block_structure=cfg.block_structure,
    block_structure_init=cfg.block_structure_init,
    model=inf_model,
    network_params=params,
    optax_solver=precond_optimizer,
    trainstate_solver=state.tx,
    preconditioner_update_steps=cfg.precond_steps,
    precond_rank = cfg.precond_rank,
    precond_identity_scaling = cfg.precond_identity_scaling,
    batch_solve_precond=cfg.batch_solve_precond,
    multibatch=cfg.multibatch_training,
    precond_on_update=cfg.precond_on_update,
    normalize_grad_for_lqr = cfg.normalize_grad_for_lqr,
    warm_start_precond = cfg.warm_start_precond,
    damping=cfg.damping,
    allow_grad_inversion=cfg.allow_grad_inversion,
    divergence_args_index=-1,
    llqr_operator_mode=cfg.llqr_operator_mode,
    llqr_checkpoint_policy=cfg.llqr_checkpoint_policy,
    llqr_use_fast_paths=cfg.llqr_use_fast_paths,
    llqr_batch_update_mode=cfg.llqr_batch_update_mode,
    llqr_batch_update_chunk_size=cfg.llqr_batch_update_chunk_size,
    llqr_second_order_mode=cfg.llqr_second_order_mode,
    llqr_second_order_chunk_size=cfg.llqr_second_order_chunk_size,
    seq2seq_preconditioner_shape_contract=ds_info.get("preconditioner_shape_contract"),
    optax_solver_requires_value_and_grad=utl.precond_solver_requires_value_and_grad(cfg.precond_solver),
  )
  llqr_batch_update_gate = preconditioner.describe_llqr_batch_update_gate()
  print(f"LLQR batch update gate: {llqr_batch_update_gate}")
  run["llqr_batch_update_gate"] = llqr_batch_update_gate
  if load_from_preexisting_model_state and precond_blocks is not None:
    preconditioner.load_blocks(precond_blocks)
    del precond_blocks

  # ---------------------------------------------------------------------------------
  # Training loop
  # ---------------------------------------------------------------------------------

  # @jax.jit
  def loss_fn(params, apply_fn, _batch_stats, x, y, _dropout_key):
    # Pass both params and batch_stats, and mark batch_stats as mutable.
    (log_probs, new_model_state) = apply_fn(
      {'params': params, 'batch_stats': _batch_stats},
      x,
      rngs={'dropout': _dropout_key},
      mutable=['batch_stats']
    )
    loss = cross_entropy_loss(
      log_probs,
      y,
      label_smoothing=cfg.label_smoothing,
      ignore_index=translation_target_pad_id,
    )
    return loss, new_model_state

  @jax.jit
  def compute_updates(params, _batch_stats, x, y, _dropout_key):
    """Compute standard gradient of the cross entropy loss."""
    return jax.value_and_grad(loss_fn, argnums=0, has_aux=True)(params, model.apply, _batch_stats, x, y, _dropout_key)

  signal.signal(signal.SIGTERM, utl.signal_handler)  # Before getting pre-empted and requeued.
  signal.signal(signal.SIGUSR1, utl.signal_handler)  # Before reaching the end of the time limit.

  # # @jax.jit
  # def train_step(state, train_dataloader, _dropout_key):
  #   x, y = next(train_dataloader)
  #   _dropout_key, consumed_key = jax.random.split(_dropout_key)
  #   (running_loss, new_model_state), running_grads = compute_updates(state.params, state.batch_stats, x, y, _dropout_key)
  #   for i in range(cfg.grad_acc_steps - 1):
  #     x, y = next(train_dataloader)
  #     _dropout_key, consumed_key = jax.random.split(_dropout_key)
  #     (loss, new_model_state), grads = compute_updates(state.params, state.batch_stats, x, y, _dropout_key)
  #     running_grads = jax.tree_util.tree_map(jnp.add, running_grads, grads)
  #     running_loss += loss
  #   running_loss /= cfg.grad_acc_steps
  #   running_grads = jax.tree_util.tree_map(lambda v : v/cfg.grad_acc_steps, running_grads)
  #   #TODO might want to revisit new_model_state handling with grad accumulation, no impact with current configs however
  #
  #   if cfg.precond_on_update:
  #     new_state = state.apply_gradients_and_precond(grads=running_grads, precond_apply=preconditioner.apply,
  #                                                   normalize_conv_params=cfg.normalize_conv_params,
  #                                                   batch_stats=new_model_state['batch_stats'])
  #   else:
  #     # Apply the preconditioner on the gradient
  #     precond_grads = preconditioner.apply(running_grads)
  #     # print(utl.pytree_l2_norm(precond_grads) - utl.pytree_l2_norm(running_grads))
  #     new_state = state.apply_gradients(grads=precond_grads, normalize_conv_params=cfg.normalize_conv_params,
  #                                       batch_stats=new_model_state['batch_stats'])
  #
  #   return new_state, running_loss, x, y

  def precond_apply_fn(blocks, grads):
    return preconditioner.apply(blocks, grads)

  def accumulate_grads(params_for_grad, batch_stats_init, key_in, x_acc, y_acc):
    """Accumulate loss/grads over x_acc, y_acc at fixed params_for_grad."""
    acc_steps = utl.tree_axis0_size(x_acc)
    key, subkey0 = jax.random.split(key_in)
    (loss0, new_model_state0), grads0 = compute_updates(
      params_for_grad, batch_stats_init, utl.tree_axis0_take(x_acc, 0), utl.tree_axis0_take(y_acc, 0), subkey0
    )

    def body(carry, inp):
      sum_loss, sum_grads, batch_stats, key = carry
      x, y = inp
      key, subkey = jax.random.split(key)

      (loss, new_model_state), grads = compute_updates(
        params_for_grad, batch_stats, x, y, subkey
      )

      sum_loss = sum_loss + loss
      sum_grads = jax.tree_map(jnp.add, sum_grads, grads)
      batch_stats = new_model_state["batch_stats"]
      return (sum_loss, sum_grads, batch_stats, key), None

    init = (loss0, grads0, new_model_state0["batch_stats"], key)
    (sum_loss, sum_grads, final_batch_stats, key_out), _ = jax.lax.scan(
      body, init, (utl.tree_axis0_slice(x_acc, 1), utl.tree_axis0_slice(y_acc, 1))
    )

    mean_loss = sum_loss / acc_steps
    mean_grads = jax.tree_map(lambda v: v / acc_steps, sum_grads)
    return mean_loss, mean_grads, final_batch_stats, key_out

  def apply_configured_training_update(state, precond_blocks, grads, batch_stats):
    if cfg.precond_on_update:
      return state.apply_gradients_and_precond(
        grads=grads,
        precond_apply=lambda g: precond_apply_fn(precond_blocks, g),
        normalize_conv_params=cfg.normalize_conv_params,
        batch_stats=batch_stats,
      )

    precond_grads = precond_apply_fn(precond_blocks, grads)
    return state.apply_gradients(
      grads=precond_grads,
      normalize_conv_params=cfg.normalize_conv_params,
      batch_stats=batch_stats,
    )

  def apply_vanilla_training_update(state, precond_blocks, grads, batch_stats):
    del precond_blocks
    return state.apply_gradients(
      grads=grads,
      normalize_conv_params=cfg.normalize_conv_params,
      batch_stats=batch_stats,
    )

  train_step_jit = sam_mode_handlers.build_train_step_jit(
    cfg,
    accumulate_grads=accumulate_grads,
    apply_training_update=apply_configured_training_update,
    apply_vanilla_training_update=apply_vanilla_training_update,
    precond_apply_fn=precond_apply_fn,
  )

  def train_step(state, precond_blocks, train_dataloader, dropout_key, step):
    x_acc, y_acc = utl.next_accumulated_batches(train_dataloader, cfg.grad_acc_steps)

    new_state, mean_loss, dropout_key = train_step_jit(
      state, precond_blocks, x_acc, y_acc, dropout_key, step=step
    )

    return new_state, mean_loss, utl.tree_axis0_take(x_acc, -1), utl.tree_axis0_take(y_acc, -1)

  # Start timer
  start_time = time.time()
  # We'll keep a local dataloader iterator

  dropout_key = jax.random.PRNGKey(cfg.rng_seed)
  if load_from_preexisting_model_state:
    dropout_key = run_state["dropout_key"]
    starting_step = run_state["training_step"]
    best_metric_name = run_state["best_metric_name"]
    best_metric_value = run_state["best_metric_value"]
    best_metric_step = run_state["best_metric_step"]
    prev_elapsed_time = run_state["training_time"]
  else:
    starting_step = 0
    best_metric_name = "accuracy"
    best_metric_value = 0.0
    best_metric_step = -1
    prev_elapsed_time = 0
  if is_translation_task and cfg.translation_eval.enabled and best_metric_name == "accuracy" and best_metric_value == 0.0:
    best_metric_name = cfg.translation_eval.checkpoint_metric
    best_metric_value = float("-inf") if cfg.translation_eval.maximize_checkpoint_metric else float("inf")
    best_metric_step = -1
    if cfg.preempt_handling:
      utl.update_run_state_best_metric(
        run_state,
        metric_name=best_metric_name,
        metric_value=best_metric_value,
        step=best_metric_step,
      )
  logged_first_batch_update_route = False
  last_logged_batch_update_route = None

  print(f"Continuing training from step {starting_step}")
  for step in range(starting_step, total_steps):
    # First, checkpoint if required
    if (step > 0) and cfg.preempt_handling and (step % cfg.checkpoint_freq == 0) and not load_from_preexisting_model_state:
      chckpt_init_time = time.time()
      elapsed_time = time.time() - start_time + prev_elapsed_time
      utl.checkpoint_exp(
        run_state,
        state,
        preconditioner.expose_blocks(),
        curr_epoch=step // steps_per_epoch_rounded,
        curr_step=step,
        dropout_key=dropout_key,
        training_time=elapsed_time,
        best_metric_name=best_metric_name,
        best_metric_value=best_metric_value,
        best_metric_step=best_metric_step,
      )
      print(
        f"Checkpointing performed in: {timedelta(seconds=time.time() - chckpt_init_time)}")

    # Run due eval before any same-step preconditioner or parameter update.
    token_eval_due = should_run_periodic_event(
      step=step,
      total_steps=total_steps,
      every=cfg.test_eval_freq,
    )
    bleu_eval_mode = None
    if is_translation_task:
      bleu_eval_mode = translation_bleu_eval_mode(
        step=step,
        total_steps=total_steps,
        enabled=bool(cfg.translation_eval.enabled),
        freq=cfg.translation_eval.freq,
        test_eval_freq=cfg.test_eval_freq,
        full_eval_at_end=bool(cfg.translation_eval.full_eval_at_end),
        full_eval_freq=cfg.translation_eval.full_eval_freq,
      )
    if token_eval_due or bleu_eval_mode is not None:
      test_time_start = time.time()
      elapsed_time = time.time() - start_time + prev_elapsed_time
      if is_translation_task:
        if token_eval_due:
          if cfg.record_histograms:
            valid_accuracy, valid_loss = compute_accuracy_and_loss_with_hists(
              state,
              test_dataloader,
              test_eval_target_count,
              run,
              step=step,
              prefix='valid',
              ignore_index=translation_target_pad_id,
            )
          else:
            valid_accuracy, valid_loss = compute_accuracy_and_loss(
              state,
              test_dataloader,
              test_eval_target_count,
              ignore_index=translation_target_pad_id,
            )
          track_translation_token_validation_metrics(
            run,
            valid_accuracy=valid_accuracy,
            valid_loss=valid_loss,
            step=step,
            elapsed_time=elapsed_time,
          )
          print("============================")
          print(f"Step {step} | Valid Loss: {valid_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
          print(f"Step {step} | Valid Token Accuracy: {valid_accuracy:.2f}%")
          print(f"Valid token evaluation computed in {time.time() - test_time_start:.2f} seconds")
          print("============================")

        if bleu_eval_mode is not None:
          bleu_time_start = time.time()
          bleu_is_full = bleu_eval_mode == "full"
          max_bleu_examples = None if bleu_is_full else cfg.translation_eval.max_examples
          metric_name = "valid bleu" if bleu_is_full else "valid bleu_sampled"
          progress_freq = cfg.translation_eval.progress_freq
          limit_msg = "full validation set" if max_bleu_examples is None else f"up to {int(max_bleu_examples)} examples"
          print("============================")
          print(
            f"Step {step} | Starting {bleu_eval_mode} BLEU evaluation "
            f"({limit_msg}, beam={cfg.translation_eval.beam_size})"
          )

          def _print_bleu_progress(example_count):
            print(
              f"Step {step} | {bleu_eval_mode} BLEU progress: "
              f"{example_count} examples | Elapsed {time.time() - bleu_time_start:.2f} seconds"
            )

          translation_metrics = evaluate_translation_generation(
            dataloader=test_dataloader,
            next_log_probs_fn=build_translation_next_log_probs_fn(model=inf_model, state=state),
            target_dictionary=translation_eval_assets["target_dictionary"],
            target_lang=cfg.dataset.target_lang,
            beam_size=cfg.translation_eval.beam_size,
            max_len_a=cfg.translation_eval.max_len_a,
            max_len_b=cfg.translation_eval.max_len_b,
            pad_id=int(ds_info["model_init_kwargs"]["pad_id"]),
            bos_id=int(ds_info["model_init_kwargs"]["bos_id"]),
            eos_id=int(ds_info["model_init_kwargs"]["eos_id"]),
            max_target_positions=int(model_kwargs.get("max_target_positions", 1024)),
            remove_bpe=cfg.translation_eval.remove_bpe,
            detok=cfg.translation_eval.detok,
            max_examples=max_bleu_examples,
            progress_freq=progress_freq,
            progress_callback=_print_bleu_progress,
          )
          valid_bleu = translation_metrics["bleu"]
          run.track(valid_bleu, name=metric_name, step=step)
          run.track(valid_bleu, name=f"{metric_name}|t", step=elapsed_time*100)
          run.track(translation_metrics["num_examples"], name=f"{metric_name} examples", step=step)
          if bleu_is_full:
            improved = utl.metric_improved(
              valid_bleu,
              best_metric_value,
              maximize=cfg.translation_eval.maximize_checkpoint_metric,
            )
            if improved:
              best_metric_name = cfg.translation_eval.checkpoint_metric
              best_metric_value = valid_bleu
              best_metric_step = step
              if cfg.preempt_handling:
                utl.update_run_state_best_metric(
                  run_state,
                  metric_name=best_metric_name,
                  metric_value=best_metric_value,
                  step=best_metric_step,
                )
                utl.save_run_state(run_state)
                utl.checkpoint_exp(
                  run_state,
                  state,
                  preconditioner.expose_blocks(),
                  curr_epoch=step // steps_per_epoch_rounded,
                  curr_step=step,
                  dropout_key=dropout_key,
                  training_time=elapsed_time,
                  target_dir=Path(run_state["model_dir"]) / "checkpoint_best",
                )
            run.track(best_metric_value, name="best valid bleu", step=step)
            run.track(best_metric_step, name="best valid bleu step", step=step)
          print(f"Step {step} | Valid BLEU ({bleu_eval_mode}): {valid_bleu:.2f}")
          print(f"Step {step} | Valid BLEU Examples ({bleu_eval_mode}): {translation_metrics['num_examples']}")
          if cfg.translation_eval.print_samples and translation_metrics["sample_hypothesis"] is not None:
            print(f"Sample Hypothesis: {translation_metrics['sample_hypothesis']}")
            print(f"Sample Reference: {translation_metrics['sample_reference']}")
          if bleu_is_full:
            print(
              f"{best_metric_name} best checkpoint metric after step {step}: "
              f"{best_metric_value:.2f} (step {best_metric_step})"
            )
          print(f"Valid {bleu_eval_mode} BLEU evaluation computed in {time.time() - bleu_time_start:.2f} seconds")
          print("============================")
      else:
        if cfg.record_histograms:
          test_accuracy, test_loss = compute_accuracy_and_loss_with_hists(state, test_dataloader, test_eval_target_count, run,
                                                                          step=step, prefix='test')
          _, _ = compute_accuracy_and_loss_with_hists(state, dataloader, train_eval_target_count, run,
                                                                          step=step, prefix='train')
        else:
          test_accuracy, test_loss = compute_accuracy_and_loss(state, test_dataloader, test_eval_target_count)
        run.track(test_accuracy, name="test accuracy", step=step)
        run.track(test_loss, name="test loss", step=step)
        run.track(test_accuracy, name="test accuracy|t", step=elapsed_time*100)
        run.track(test_loss, name="test loss|t", step=elapsed_time*100)
        if cfg.add_train_eval:
          if cfg.record_histograms:
            train_eval_accuracy, train_eval_loss = compute_accuracy_and_loss_with_hists(state, train_eval_dataloader,
                                                                                        num_samples=train_eval_target_count,
                                                                                        run=run,
                                                                                        step=step, prefix='train_eval')
          else:
            train_eval_accuracy, train_eval_loss = compute_accuracy_and_loss(state, train_eval_dataloader,
                                                                             num_samples=train_eval_target_count)
          run.track(train_eval_accuracy, name="train_eval accuracy", step=step)
          run.track(train_eval_loss, name="train_eval loss", step=step)
        print("============================")
        print(f"Step {step} | Test Loss: {test_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Step {step} | Test Accuracy: {test_accuracy:.2f}%")
        print(f"Test accuracy across entire dataset computed in {time.time() - test_time_start:.2f} seconds")
        print("============================")

    if not should_run_training_step(step=step, total_steps=total_steps):
      continue

    # Possibly update the preconditioner every `update_preconditioner_every` steps
    _update_precond_every = precond_up_sched(step)
    if load_from_preexisting_model_state:
      # trigger compilation
      precond_lr = precond_lr_fn(step)
      preconditioner.compile_precond_updater(state.params, dataloader, precond_lr, state.opt_state, cfg.precond_batch_size,
                                           other_model_variables={'batch_stats': state.batch_stats})
      compile_route = preconditioner.describe_last_llqr_batch_update_route()
      print(f"Compiled LLQR preconditioner update route: {compile_route}")
      run["llqr_batch_compile_route"] = compile_route
      load_from_preexisting_model_state = False
    if (step % _update_precond_every) == 0 and cfg.use_preconditioner and step < cfg.update_preconditioner_until:
      # The preconditioner update can be run on a mini-batch from the dataloader
      # We do multiple steps (precond_steps) of "preconditioner training"
      precond_update_start_time = time.time()
      precond_lr = precond_lr_fn(step)
      _ema_decay = ema_fn(step)
      preconditioner.update_preconditioner(state.params, dataloader, precond_lr, state.opt_state, cfg.precond_batch_size, _ema_decay,
                                           other_model_variables={'batch_stats': state.batch_stats})
      update_route = preconditioner.describe_last_llqr_batch_update_route()
      if not logged_first_batch_update_route:
        print(f"LLQR preconditioner update route at step {step}: {update_route}")
        run["llqr_batch_first_update_route"] = update_route
        logged_first_batch_update_route = True
      if last_logged_batch_update_route != update_route:
        if logged_first_batch_update_route and last_logged_batch_update_route is not None:
          print(f"LLQR preconditioner update route changed at step {step}: {update_route}")
        run["llqr_batch_last_update_route"] = update_route
        last_logged_batch_update_route = dict(update_route)
      precond_max, precond_min, precond_norm, per_layer_norm = preconditioner.get_stats()
      # !!Remove below when timing against non-2nd order methods!! (Affect computation time)
      run.track(precond_max, name="Maximum across preconditioner", step=step)
      run.track(precond_min, name="Minimum across preconditioner", step=step)
      run.track(precond_norm, name="Preconditioner l2 norm", step=step)
      for layer, l_norm in per_layer_norm.items():
        run.track(l_norm, name=f"{layer} l2 norm", step=step)
      print(f"Preconditioner was updated in {time.time()-precond_update_start_time:.2f} seconds")

      # Asymmetry check
      if cfg.measure_asymmetry:
        skews_dict = preconditioner.get_precond_asymmetry()
        for layer, layer_skews in skews_dict.items():
          # layer_skews: List[Tuple[frob, spectral]]
          if len(layer_skews) == 0:
            continue

          if len(layer_skews) == 1:
            frob_skew, spectral_skew = layer_skews[0]
            run.track(frob_skew, name=f"{layer}/frob skew", step=step)
            run.track(spectral_skew, name=f"{layer}/spectral skew", step=step)
          else:
            for i, (frob_skew, spectral_skew) in enumerate(layer_skews, start=1):
              run.track(
                frob_skew,
                name=f"{layer}/part {i}/frob skew",
                step=step,
              )
              run.track(
                spectral_skew,
                name=f"{layer}/part {i}/spectral skew",
                step=step,
              )
        avg_frob_skew, avg_spectral_skew = utl.average_skews(skews_dict)
        run.track(avg_frob_skew, name="Average frob skew across preconditioner", step=step)
        run.track(avg_spectral_skew, name="Average spectral skew across preconditioner", step=step)

    # Grab the next batch for normal training
    # x_batch, y_batch = next(dataloader)
    #
    dropout_key, consumed_key = jax.random.split(dropout_key)
    state, loss, x_batch, y_batch = train_step(state, preconditioner.expose_blocks(), dataloader, consumed_key, step)

    # Logging or testing every so often
    if should_run_periodic_event(step=step, total_steps=total_steps, every=cfg.logging_freq):
      # Simple logging
      # train_loss = loss_eval(state, x_batch, y_batch)
      train_loss = loss
      elapsed_time = time.time() - start_time + prev_elapsed_time  # Calculate elapsed time
      run.track(train_loss, name="train loss", step=step)
      run.track(train_loss, name="train loss|t", step=elapsed_time*100)
      # Compute batch accuracy
      batch_accuracy, _ = compute_batch_accuracy(
        state,
        x_batch,
        y_batch,
        ignore_index=translation_target_pad_id,
      )
      train_accuracy_name = "train token_accuracy" if is_translation_task else "train accuracy"
      run.track(batch_accuracy, name=train_accuracy_name, step=step)
      run.track(batch_accuracy, name=f"{train_accuracy_name}|t", step=elapsed_time*100)
      if should_run_periodic_event(step=step, total_steps=total_steps, every=cfg.report_freq):
        # Print info
        print(f"Step {step} | Train Loss: {train_loss:.4f} | Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Step {step} | Batch Accuracy: {batch_accuracy:.2f}%")


  print("Training complete!")
  # End timer
  total_time = time.time() - start_time + prev_elapsed_time
  print(f"Training complete! Total Time Elapsed: {total_time:.2f} seconds")


if __name__ == "__main__":
  main()
