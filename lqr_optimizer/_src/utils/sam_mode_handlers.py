import jax

import lqr_optimizer._src.utils.utils as utl


SUPPORTED_SAM_MODES = (None, "base_sam", "base_fsam", "past_fsam")


def _normalize_sam_mode(sam_mode):
  return None if not sam_mode else sam_mode


def validate_sam_mode_contract(cfg, opt_state):
  sam_mode = _normalize_sam_mode(cfg.sam_mode)
  if sam_mode not in SUPPORTED_SAM_MODES:
    raise ValueError(f"Unsupported sam_mode: {cfg.sam_mode}")
  utl.validate_sam_research_contract(cfg, opt_state)


def build_train_step_jit(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    precond_apply_fn,
):
  sam_mode = _normalize_sam_mode(cfg.sam_mode)
  builders = {
    None: _build_no_sam_train_step,
    "base_sam": _build_base_sam_train_step,
    "past_fsam": _build_past_fsam_train_step,
    "base_fsam": _build_base_fsam_train_step,
  }
  try:
    builder = builders[sam_mode]
  except KeyError as exc:
    raise ValueError(f"Unsupported sam_mode: {cfg.sam_mode}") from exc
  return builder(
    cfg,
    accumulate_grads=accumulate_grads,
    apply_training_update=apply_training_update,
    precond_apply_fn=precond_apply_fn,
  )


def _build_no_sam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    precond_apply_fn,
):
  del cfg, precond_apply_fn

  @jax.jit
  def train_step_jit(state, precond_blocks, x_acc, y_acc, dropout_key):
    mean_loss, mean_grads, final_batch_stats, key_out = accumulate_grads(
      state.params, state.batch_stats, dropout_key, x_acc, y_acc
    )
    new_state = apply_training_update(
      state, precond_blocks, mean_grads, final_batch_stats
    )
    return new_state, mean_loss, key_out

  return train_step_jit


def _build_base_sam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    precond_apply_fn,
):
  @jax.jit
  def train_step_jit(state, precond_blocks, x_acc, y_acc, dropout_key):
    mean_loss_center, mean_grads_center, _, key_after_center = accumulate_grads(
      state.params, state.batch_stats, dropout_key, x_acc, y_acc
    )

    perturbation_vector, key_for_perturbed_pass = (
      utl.resolve_sam_ablation_perturbation_vector(
        sam_mode="base_sam",
        base_vector_source=cfg.sam_research_base_vector_source,
        mean_grads_center=mean_grads_center,
        g_bar=state.gbar,
        opt_state=state.opt_state,
        rng_key=key_after_center,
      )
    )
    eps_tree = utl.make_perturbation_from_vector(
      precond_blocks=precond_blocks,
      vector=perturbation_vector,
      precond_apply_fn=precond_apply_fn,
      rho=cfg.perturbation_rho,
      mode=cfg.perturb_mode,
      norm_mode=cfg.norm_mode,
      eps=cfg.gbar_eps,
    )
    eps_tree = utl.apply_sam_perturb_sign(
      eps_tree,
      cfg.sam_research_perturb_sign,
    )

    params_pert = utl.tree_add(state.params, eps_tree)
    mean_loss_pert, mean_grads_pert, final_batch_stats, key_out = accumulate_grads(
      params_pert, state.batch_stats, key_for_perturbed_pass, x_acc, y_acc
    )

    new_state = apply_training_update(
      state, precond_blocks, mean_grads_pert, final_batch_stats
    )
    new_gbar, new_g_last = utl.resolve_sam_state_buffers(
      sam_mode="base_sam",
      g_bar=state.gbar,
      g_last=state.g_last,
      mean_grads_center=mean_grads_center,
      mean_grads_pert=mean_grads_pert,
      precond_blocks=precond_blocks,
      precond_apply_fn=precond_apply_fn,
      beta=cfg.gbar_beta,
      mode=cfg.perturb_mode,
      eps=cfg.gbar_eps,
    )
    new_state = new_state.replace(gbar=new_gbar, g_last=new_g_last)
    return new_state, mean_loss_pert, key_out

  return train_step_jit


def _build_past_fsam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    precond_apply_fn,
):
  @jax.jit
  def train_step_jit(state, precond_blocks, x_acc, y_acc, dropout_key):
    eps_tree = utl.make_perturbation_from_noise(
      precond_blocks=precond_blocks,
      g_last=state.g_last,
      g_bar=state.gbar,
      precond_apply_fn=precond_apply_fn,
      rho=cfg.perturbation_rho,
      mode=cfg.perturb_mode,
      norm_mode=cfg.norm_mode,
      eps=cfg.gbar_eps,
    )

    params_pert = utl.tree_add(state.params, eps_tree)
    mean_loss, mean_grads, final_batch_stats, key_out = accumulate_grads(
      params_pert, state.batch_stats, dropout_key, x_acc, y_acc
    )

    new_state = apply_training_update(
      state, precond_blocks, mean_grads, final_batch_stats
    )
    new_gbar, new_g_last = utl.resolve_sam_state_buffers(
      sam_mode="past_fsam",
      g_bar=state.gbar,
      g_last=state.g_last,
      mean_grads_center=None,
      mean_grads_pert=mean_grads,
      precond_blocks=precond_blocks,
      precond_apply_fn=precond_apply_fn,
      beta=cfg.gbar_beta,
      mode=cfg.perturb_mode,
      eps=cfg.gbar_eps,
    )
    new_state = new_state.replace(
      gbar=new_gbar,
      g_last=new_g_last,
    )
    return new_state, mean_loss, key_out

  return train_step_jit


def _build_base_fsam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    precond_apply_fn,
):
  @jax.jit
  def train_step_jit(state, precond_blocks, x_acc, y_acc, dropout_key):
    mean_loss_center, mean_grads_center, _, key_after_center = accumulate_grads(
      state.params, state.batch_stats, dropout_key, x_acc, y_acc
    )

    perturbation_vector, key_for_perturbed_pass = (
      utl.resolve_sam_ablation_perturbation_vector(
        sam_mode="base_fsam",
        base_vector_source=cfg.sam_research_base_vector_source,
        mean_grads_center=mean_grads_center,
        g_bar=state.gbar,
        opt_state=state.opt_state,
        rng_key=key_after_center,
      )
    )
    eps_tree = utl.make_perturbation_from_vector(
      precond_blocks=precond_blocks,
      vector=perturbation_vector,
      precond_apply_fn=precond_apply_fn,
      rho=cfg.perturbation_rho,
      mode=cfg.perturb_mode,
      norm_mode=cfg.norm_mode,
      eps=cfg.gbar_eps,
    )
    eps_tree = utl.apply_sam_perturb_sign(
      eps_tree,
      cfg.sam_research_perturb_sign,
    )

    params_pert = utl.tree_add(state.params, eps_tree)
    mean_loss_pert, mean_grads_pert, final_batch_stats, key_out = accumulate_grads(
      params_pert, state.batch_stats, key_for_perturbed_pass, x_acc, y_acc
    )

    new_state = apply_training_update(
      state, precond_blocks, mean_grads_pert, final_batch_stats
    )
    new_gbar, new_g_last = utl.resolve_sam_state_buffers(
      sam_mode="base_fsam",
      g_bar=state.gbar,
      g_last=state.g_last,
      mean_grads_center=mean_grads_center,
      mean_grads_pert=mean_grads_pert,
      precond_blocks=precond_blocks,
      precond_apply_fn=precond_apply_fn,
      beta=cfg.gbar_beta,
      mode=cfg.perturb_mode,
      eps=cfg.gbar_eps,
    )
    new_state = new_state.replace(gbar=new_gbar, g_last=new_g_last)
    return new_state, mean_loss_pert, key_out

  return train_step_jit
