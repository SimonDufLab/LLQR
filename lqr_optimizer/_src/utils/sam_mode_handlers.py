import jax

import lqr_optimizer._src.utils.utils as utl


SUPPORTED_SAM_MODES = (None, "base_sam", "base_fsam", "past_fsam", "asam", "fisher_sam")
_LEGACY_SAM_NEUTRAL_DEFAULTS = (
  ("perturb_mode", "ema_grad"),
  ("norm_mode", "euclidean"),
  # ("sam_research_base_vector_source", "current_gradient"), # overkill limitations, can work with those, even if not well-formed
  # ("sam_research_perturb_sign", "ascent"),
  # ("gbar_beta", 0.9),
  # ("gbar_eps", 1e-12),
)


def _normalize_sam_mode(sam_mode):
  return None if not sam_mode else sam_mode


def _sam_outer_update_uses_preconditioner(cfg):
  return getattr(cfg, "sam_use_preconditioner_on_update", True)


def _llqr_outer_update_toggle_is_active(sam_mode):
  return sam_mode in ("base_sam", "base_fsam", "past_fsam", "asam")


def _resolve_sam_outer_update_apply_fn(
    cfg,
    *,
    sam_mode,
    apply_training_update,
    apply_vanilla_training_update,
):
  if sam_mode is None:
    return apply_training_update
  if sam_mode == "fisher_sam":
    return apply_vanilla_training_update
  if _sam_outer_update_uses_preconditioner(cfg):
    return apply_training_update
  return apply_vanilla_training_update


def _validate_mode_uses_neutral_legacy_defaults(cfg, *, sam_mode):
  for field_name, expected_value in _LEGACY_SAM_NEUTRAL_DEFAULTS:
    actual_value = getattr(cfg, field_name)
    if actual_value != expected_value:
      raise ValueError(
        f"sam_mode={sam_mode!r} requires {field_name}={expected_value!r}; got {actual_value!r}."
      )


def _validate_asam_mode_contract(cfg):
  if cfg.asam_eta < 0:
    raise ValueError(f"sam_mode='asam' requires asam_eta >= 0; got {cfg.asam_eta!r}.")

  _validate_mode_uses_neutral_legacy_defaults(cfg, sam_mode="asam")


def _validate_fisher_sam_mode_contract(cfg):
  if cfg.fisher_sam_eta < 0:
    raise ValueError(
      f"sam_mode='fisher_sam' requires fisher_sam_eta >= 0; got {cfg.fisher_sam_eta!r}."
    )

  _validate_mode_uses_neutral_legacy_defaults(cfg, sam_mode="fisher_sam")


def validate_sam_mode_contract(cfg, opt_state):
  sam_mode = _normalize_sam_mode(cfg.sam_mode)
  if sam_mode not in SUPPORTED_SAM_MODES:
    raise ValueError(f"Unsupported sam_mode: {cfg.sam_mode}")
  if sam_mode == "asam":
    _validate_asam_mode_contract(cfg)
  if sam_mode == "fisher_sam":
    _validate_fisher_sam_mode_contract(cfg)
  if (
      _llqr_outer_update_toggle_is_active(sam_mode)
      and _sam_outer_update_uses_preconditioner(cfg)
      and not getattr(cfg, "use_preconditioner", True)
  ):
    raise ValueError(
      f"sam_mode={sam_mode!r} requires use_preconditioner=true when "
      "sam_use_preconditioner_on_update=true."
    )
  utl.validate_sam_research_contract(cfg, opt_state)


def build_train_step_jit(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    apply_vanilla_training_update,
    precond_apply_fn,
):
  sam_mode = _normalize_sam_mode(cfg.sam_mode)
  builders = {
    None: _build_no_sam_train_step,
    "base_sam": _build_base_sam_train_step,
    "past_fsam": _build_past_fsam_train_step,
    "base_fsam": _build_base_fsam_train_step,
    "asam": _build_asam_train_step,
    "fisher_sam": _build_fisher_sam_train_step,
  }
  try:
    builder = builders[sam_mode]
  except KeyError as exc:
    raise ValueError(f"Unsupported sam_mode: {cfg.sam_mode}") from exc
  return builder(
    cfg,
    accumulate_grads=accumulate_grads,
    apply_training_update=apply_training_update,
    apply_vanilla_training_update=apply_vanilla_training_update,
    precond_apply_fn=precond_apply_fn,
  )


def _build_no_sam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    apply_vanilla_training_update,
    precond_apply_fn,
):
  del cfg, apply_vanilla_training_update, precond_apply_fn

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
    apply_vanilla_training_update,
    precond_apply_fn,
):
  apply_outer_update = _resolve_sam_outer_update_apply_fn(
    cfg,
    sam_mode="base_sam",
    apply_training_update=apply_training_update,
    apply_vanilla_training_update=apply_vanilla_training_update,
  )

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

    new_state = apply_outer_update(
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


def _build_asam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    apply_vanilla_training_update,
    precond_apply_fn,
):
  del precond_apply_fn
  apply_outer_update = _resolve_sam_outer_update_apply_fn(
    cfg,
    sam_mode="asam",
    apply_training_update=apply_training_update,
    apply_vanilla_training_update=apply_vanilla_training_update,
  )

  @jax.jit
  def train_step_jit(state, precond_blocks, x_acc, y_acc, dropout_key):
    mean_loss_center, mean_grads_center, _, key_after_center = accumulate_grads(
      state.params, state.batch_stats, dropout_key, x_acc, y_acc
    )

    eps_tree = utl.make_asam_perturbation_from_grad(
      params=state.params,
      grad=mean_grads_center,
      rho=cfg.perturbation_rho,
      eta=cfg.asam_eta,
    )

    params_pert = utl.tree_add(state.params, eps_tree)
    mean_loss_pert, mean_grads_pert, final_batch_stats, key_out = accumulate_grads(
      params_pert, state.batch_stats, key_after_center, x_acc, y_acc
    )

    new_state = apply_outer_update(
      state, precond_blocks, mean_grads_pert, final_batch_stats
    )
    new_gbar, new_g_last = utl.resolve_sam_state_buffers(
      sam_mode="asam",
      g_bar=state.gbar,
      g_last=state.g_last,
      mean_grads_center=mean_grads_center,
      mean_grads_pert=mean_grads_pert,
      precond_blocks=precond_blocks,
      precond_apply_fn=lambda *_args, **_kwargs: None,
      beta=cfg.gbar_beta,
      mode=cfg.perturb_mode,
      eps=cfg.gbar_eps,
    )
    new_state = new_state.replace(gbar=new_gbar, g_last=new_g_last)
    return new_state, mean_loss_pert, key_out

  return train_step_jit


def _build_fisher_sam_train_step(
    cfg,
    *,
    accumulate_grads,
    apply_training_update,
    apply_vanilla_training_update,
    precond_apply_fn,
):
  del apply_training_update, precond_apply_fn

  @jax.jit
  def train_step_jit(state, precond_blocks, x_acc, y_acc, dropout_key):
    mean_loss_center, mean_grads_center, _, key_after_center = accumulate_grads(
      state.params, state.batch_stats, dropout_key, x_acc, y_acc
    )

    eps_tree = utl.make_fisher_sam_perturbation_from_grad(
      grad=mean_grads_center,
      rho=cfg.perturbation_rho,
      eta=cfg.fisher_sam_eta,
    )

    params_pert = utl.tree_add(state.params, eps_tree)
    mean_loss_pert, mean_grads_pert, final_batch_stats, key_out = accumulate_grads(
      params_pert, state.batch_stats, key_after_center, x_acc, y_acc
    )

    new_state = apply_vanilla_training_update(
      state, precond_blocks, mean_grads_pert, final_batch_stats
    )
    new_gbar, new_g_last = utl.resolve_sam_state_buffers(
      sam_mode="fisher_sam",
      g_bar=state.gbar,
      g_last=state.g_last,
      mean_grads_center=mean_grads_center,
      mean_grads_pert=mean_grads_pert,
      precond_blocks=precond_blocks,
      precond_apply_fn=lambda *_args, **_kwargs: None,
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
    apply_vanilla_training_update,
    precond_apply_fn,
):
  apply_outer_update = _resolve_sam_outer_update_apply_fn(
    cfg,
    sam_mode="past_fsam",
    apply_training_update=apply_training_update,
    apply_vanilla_training_update=apply_vanilla_training_update,
  )

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

    new_state = apply_outer_update(
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
    apply_vanilla_training_update,
    precond_apply_fn,
):
  apply_outer_update = _resolve_sam_outer_update_apply_fn(
    cfg,
    sam_mode="base_fsam",
    apply_training_update=apply_training_update,
    apply_vanilla_training_update=apply_vanilla_training_update,
  )

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

    new_state = apply_outer_update(
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
