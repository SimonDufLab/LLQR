"""Preconditioner classes and their update rules."""
import abc
from functools import partial

import optax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.core.frozen_dict import FrozenDict, freeze

from lqr_optimizer._src.utils.utils import (normalize_gradient, timed_jit, pytree_max_min, pytree_l2_norm,
                                            get_per_layer_norm, _deep_copy_pytree, infer_batch_layout, get_per_layer_skews)
import lqr_optimizer._src.block_matrices_approx.block_structures as block_structures
from lqr_optimizer._src.utils.build_lqr import (lqr_forward_matrices_and_states,
                             lqr_active_controllable_forward_operators_and_states,
                             lqr_active_execution_forward_operators_and_states,
                             lqr_final_costs_and_adjoints, lqr_active_final_costs_and_adjoints,
                             lqr_backward_matrices_and_adjoints,
                             lqr_backward_hamiltonian_operators,
                             lqr_active_controllable_backward_hamiltonian_operators,
                             lqr_active_controllable_backward_hamiltonian_operators_lowmem,
                             lqr_active_execution_backward_hamiltonian_operators,
                             lqr_active_execution_backward_hamiltonian_operators_lowmem,
                             prepare_active_execution_stage_metadata,
                             tree_vdot)

BLOCK_STRUCTURE_DICT = {
  'dense': block_structures.DenseBlock,
  'diagonal': block_structures.DiagonalBlock,
  'scalar': block_structures.ScalarBlock,
  'kfac': block_structures.KroneckerBlock,
  'e-kfac': block_structures.EKFACBlock,
  'sep-e-kfac': block_structures.SeparableEKFACBlock,
  'psd-e-kfac': block_structures.PSDEKFACBlock,
  'psd-sep-e-kfac': block_structures.PSDSeparableEKFACBlock,
  'rms_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="rms"),
  'max_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="maxabs"),
  'scaled-rms_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="rms", use_alpha=True),
  'scaled-max_norm-e-kfac': Partial(block_structures.EKFACBlockNormalized, norm_mode="maxabs", use_alpha=True),
  'h-kfac': block_structures.HouseholderDiagKroneckerBlock,
  'diag-kfac': block_structures.DiagKroneckerBlock,
  'sym-kfac' : block_structures.MirrorSymKroneckerBlock,
  'psd-sym-kfac' : block_structures.PSDSymKroneckerBlock,
  'sds-sym-kfac' : block_structures.SDSSymKroneckerBlock,
  'sds-sym-kfac_norm-d' : block_structures.NormalizedSDSSymKroneckerBlock,
  'low-rank-memory': block_structures.LowRankBlockMemory,
  'low-rank-memory-asym': block_structures.LowRankBlockMemoryAsym,
  'sym_swm_kfac': block_structures.Sym_SWM_KFAC,
  'asym_swm_kfac': block_structures.Asym_SWM_KFAC,
  'sym_swm_e-kfac': block_structures.Sym_SWM_EKFAC,
  'sym_swm_sep-e-kfac': block_structures.Sym_SWM_SeparableEKFAC,
}


VALID_LLQR_OPERATOR_MODES = ("cached_exact", "lowmem_exact_k")


def _recover_loss_gradients_from_transition_transposes(params_or_layer_names, layer_names_or_transition_transposes,
                                                       transition_transposes_or_final_lin_cost,
                                                       final_lin_cost_or_unravel_params_fns=None,
                                                       unravel_params_fns=None, *,
                                                       first_transition_transpose=None,
                                                       freeze_result=False):
  """Recover the exact loss gradient by a reverse sweep over the joint transition transposes."""
  if isinstance(params_or_layer_names, (dict, FrozenDict)):
    params = params_or_layer_names
    layer_names = layer_names_or_transition_transposes
    transition_transposes = transition_transposes_or_final_lin_cost
    final_lin_cost = final_lin_cost_or_unravel_params_fns
    if unravel_params_fns is None:
      unravel_params_fns = {
        layer_name: ravel_pytree(params[layer_name])[1]
        for layer_name in layer_names
      }
    freeze_result = isinstance(params, FrozenDict)
  else:
    layer_names = params_or_layer_names
    transition_transposes = layer_names_or_transition_transposes
    final_lin_cost = transition_transposes_or_final_lin_cost
    unravel_params_fns = final_lin_cost_or_unravel_params_fns

  state_cotangent = final_lin_cost
  recovered_by_layer = {}

  start_index = len(layer_names) - 1
  stop_index = -1 if first_transition_transpose is None else 0
  for layer_index in range(start_index, stop_index, -1):
    layer_name = layer_names[layer_index]
    unravel_params_fn = unravel_params_fns[layer_name]
    transpose_index = layer_index if first_transition_transpose is None else layer_index - 1
    param_cotangent, state_cotangent = transition_transposes[transpose_index](state_cotangent)
    recovered_by_layer[layer_name] = unravel_params_fn(jnp.ravel(jnp.atleast_1d(param_cotangent)))

  if first_transition_transpose is not None:
    first_layer_name = layer_names[0]
    unravel_params_fn = unravel_params_fns[first_layer_name]
    param_cotangent = first_transition_transpose(state_cotangent)
    recovered_by_layer[first_layer_name] = unravel_params_fn(jnp.ravel(jnp.atleast_1d(param_cotangent)))

  ordered_recovered = {layer_name: recovered_by_layer[layer_name] for layer_name in layer_names}
  if freeze_result:
    return freeze(ordered_recovered)
  return ordered_recovered


def _materialize_active_flat_controls(block_structure, blocks, layer_names, prepared_gradients):
  """Materialize the active-path flat controls once for a fixed preconditioner tree."""
  return {
    layer_name: block_structure.train_matrix_product_flat_layer(
      blocks, layer_name, prepared_gradients[layer_name]
    )
    for layer_name in layer_names
  }


def _rollout_active_control_states(layer_names, flat_controls, first_transition, transitions):
  """Roll out the active control problem and save the state entering each later layer."""
  if not layer_names:
    raise ValueError("_rollout_active_control_states expects at least one layer")

  later_state_inputs = {}
  x = first_transition(flat_controls[layer_names[0]])
  for i, layer_name in enumerate(layer_names[1:]):
    later_state_inputs[layer_name] = x
    x = transitions[i](flat_controls[layer_name], x)

  return later_state_inputs, x


def _active_terminal_state_cotangent(final_state, final_q, final_lin_cost):
  """Recover the terminal cotangent on the active tree-native contract."""
  return jax.tree_map(jnp.add, final_lin_cost, final_q(final_state))


def _active_terminal_cost(final_state, final_q, final_lin_cost):
  """Evaluate the active terminal contribution without flattening the final state."""
  return tree_vdot(final_state, final_lin_cost) + 0.5 * tree_vdot(final_state, final_q(final_state))


def _active_lqr_control_cost(layer_names, flat_controls, first_transition, transitions,
                             first_k_backward, k_backward, final_q, final_lin_cost):
  """Evaluate the fixed-control active LLQR cost."""
  later_state_inputs, final_state = _rollout_active_control_states(
    layer_names, flat_controls, first_transition, transitions)

  return _active_lqr_control_cost_from_rollout(
    layer_names, flat_controls, later_state_inputs, final_state,
    first_k_backward, k_backward, final_q, final_lin_cost)


def _active_lqr_control_cost_from_rollout(layer_names, flat_controls, later_state_inputs, final_state,
                                          first_k_backward, k_backward, final_q, final_lin_cost):
  """Evaluate the fixed-control active LLQR cost from a precomputed rollout."""
  first_layer_name = layer_names[0]
  cost = jnp.dot(flat_controls[first_layer_name], first_k_backward(flat_controls[first_layer_name])) / 2

  for i, layer_name in enumerate(layer_names[1:]):
    u = flat_controls[layer_name]
    x = later_state_inputs[layer_name]
    k_u, k_x = k_backward[i](u, x)
    cost += (jnp.dot(u, k_u) + tree_vdot(x, k_x)) / 2

  return cost + _active_terminal_cost(final_state, final_q, final_lin_cost)


def _recover_control_gradients_from_active_lqr_adjoint(layer_names, flat_controls, later_state_inputs, final_state,
                                                       first_transition_transpose, transition_transposes,
                                                       first_k_backward, k_backward, final_q, final_lin_cost):
  """Recover exact flat control gradients by a backward sweep over the active control rollout."""
  if not layer_names:
    raise ValueError("_recover_control_gradients_from_active_lqr_adjoint expects at least one layer")

  state_cotangent = _active_terminal_state_cotangent(final_state, final_q, final_lin_cost)
  recovered_by_layer = {}

  for reverse_index in range(len(layer_names) - 2, -1, -1):
    layer_name = layer_names[reverse_index + 1]
    u = flat_controls[layer_name]
    x = later_state_inputs[layer_name]
    stage_u_grad, stage_x_grad = k_backward[reverse_index](u, x)
    control_cotangent, prev_state_cotangent = transition_transposes[reverse_index](state_cotangent)
    recovered_by_layer[layer_name] = stage_u_grad + control_cotangent
    state_cotangent = jax.tree_map(jnp.add, stage_x_grad, prev_state_cotangent)

  first_layer_name = layer_names[0]
  recovered_by_layer[first_layer_name] = (
    first_k_backward(flat_controls[first_layer_name]) + first_transition_transpose(state_cotangent)
  )

  return {layer_name: recovered_by_layer[layer_name] for layer_name in layer_names}


def _recover_preconditioner_gradients_from_active_control_adjoint(block_structure, blocks, layer_names,
                                                                  prepared_gradients, operators):
  """Recover exact preconditioner gradients from the active control adjoint."""
  (first_transition, first_transition_transpose, transitions, transition_transposes,
   first_k_backward, k_backward, final_q, final_lin_cost) = operators
  flat_controls = _materialize_active_flat_controls(
    block_structure, blocks, layer_names, prepared_gradients)
  later_state_inputs, final_state = _rollout_active_control_states(
    layer_names, flat_controls, first_transition, transitions)
  control_gradients = _recover_control_gradients_from_active_lqr_adjoint(
    layer_names, flat_controls, later_state_inputs, final_state,
    first_transition_transpose, transition_transposes,
    first_k_backward, k_backward, final_q, final_lin_cost)

  recovered_by_layer = {
    layer_name: block_structure.preconditioner_param_pullback_flat_layer(
      blocks, layer_name, prepared_gradients[layer_name], control_gradients[layer_name]
    )
    for layer_name in layer_names
  }

  if isinstance(blocks, FrozenDict):
    return freeze(recovered_by_layer)
  return recovered_by_layer


def _active_preconditioner_value_and_grad_from_control_adjoint(block_structure, blocks, layer_names,
                                                               prepared_gradients, operators):
  """Evaluate the active LLQR objective and exact preconditioner gradient without global reverse-mode AD."""
  (first_transition, first_transition_transpose, transitions, transition_transposes,
   first_k_backward, k_backward, final_q, final_lin_cost) = operators
  flat_controls = _materialize_active_flat_controls(
    block_structure, blocks, layer_names, prepared_gradients)
  later_state_inputs, final_state = _rollout_active_control_states(
    layer_names, flat_controls, first_transition, transitions)
  value = _active_lqr_control_cost_from_rollout(
    layer_names, flat_controls, later_state_inputs, final_state,
    first_k_backward, k_backward, final_q, final_lin_cost)
  control_gradients = _recover_control_gradients_from_active_lqr_adjoint(
    layer_names, flat_controls, later_state_inputs, final_state,
    first_transition_transpose, transition_transposes,
    first_k_backward, k_backward, final_q, final_lin_cost)

  recovered_by_layer = {
    layer_name: block_structure.preconditioner_param_pullback_flat_layer(
      blocks, layer_name, prepared_gradients[layer_name], control_gradients[layer_name]
    )
    for layer_name in layer_names
  }

  if isinstance(blocks, FrozenDict):
    recovered_by_layer = freeze(recovered_by_layer)
  return value, recovered_by_layer


def _recover_loss_gradients_from_execution_stage_transposes(params_or_controlled_stage_names,
                                                            controlled_stage_names_or_execution_stage_specs,
                                                            execution_stage_specs_or_execution_stage_operators,
                                                            execution_stage_operators_or_final_lin_cost,
                                                            final_lin_cost_or_unravel_params_fns,
                                                            unravel_params_fns=None,
                                                            freeze_result=False):
  if isinstance(params_or_controlled_stage_names, (dict, FrozenDict)):
    params = params_or_controlled_stage_names
    controlled_stage_names = controlled_stage_names_or_execution_stage_specs
    execution_stage_specs = execution_stage_specs_or_execution_stage_operators
    execution_stage_operators = execution_stage_operators_or_final_lin_cost
    final_lin_cost = final_lin_cost_or_unravel_params_fns
    if unravel_params_fns is None:
      unravel_params_fns = {
        layer_name: ravel_pytree(params[layer_name])[1]
        for layer_name in controlled_stage_names
      }
    freeze_result = isinstance(params, FrozenDict)
  else:
    controlled_stage_names = params_or_controlled_stage_names
    execution_stage_specs = controlled_stage_names_or_execution_stage_specs
    execution_stage_operators = execution_stage_specs_or_execution_stage_operators
    final_lin_cost = execution_stage_operators_or_final_lin_cost
    unravel_params_fns = final_lin_cost_or_unravel_params_fns

  state_cotangent = final_lin_cost
  recovered_by_layer = {}

  for stage_spec, stage_operator in zip(reversed(execution_stage_specs), reversed(execution_stage_operators)):
    if stage_operator.kind == "passive":
      state_cotangent = stage_operator.transpose(state_cotangent)
      continue

    unravel_params_fn = unravel_params_fns[stage_spec.param_name]
    if stage_operator.kind == "control_only":
      param_cotangent = stage_operator.transpose(state_cotangent)
      recovered_by_layer[stage_spec.param_name] = unravel_params_fn(jnp.ravel(jnp.atleast_1d(param_cotangent)))
      break

    param_cotangent, state_cotangent = stage_operator.transpose(state_cotangent)
    recovered_by_layer[stage_spec.param_name] = unravel_params_fn(jnp.ravel(jnp.atleast_1d(param_cotangent)))

  ordered_recovered = {layer_name: recovered_by_layer[layer_name] for layer_name in controlled_stage_names}
  if freeze_result:
    return freeze(ordered_recovered)
  return ordered_recovered


def _rollout_active_execution_stages(execution_stage_specs, execution_stage_operators, flat_controls):
  stage_inputs = {}
  x = None
  for stage_spec, stage_operator in zip(execution_stage_specs, execution_stage_operators):
    if stage_operator.kind == "control_only":
      stage_inputs[stage_spec.name] = None
      x = stage_operator.forward(flat_controls[stage_spec.param_name])
    elif stage_operator.kind == "controlled":
      if x is None:
        raise ValueError("Encountered a controlled execution stage before the first control-only stage.")
      stage_inputs[stage_spec.name] = x
      x = stage_operator.forward(flat_controls[stage_spec.param_name], x)
    else:
      if x is None:
        raise ValueError("Passive execution stages before the first controlled stage are not supported.")
      stage_inputs[stage_spec.name] = x
      x = stage_operator.forward(x)
  return stage_inputs, x


def _active_execution_lqr_control_cost_from_rollout(execution_stage_specs, execution_stage_operators, stage_k_operators,
                                                    flat_controls, stage_inputs, final_state, final_q, final_lin_cost):
  cost = 0.0
  for stage_spec, stage_operator, stage_k in zip(execution_stage_specs, execution_stage_operators, stage_k_operators):
    if stage_operator.kind == "control_only":
      u = flat_controls[stage_spec.param_name]
      cost += jnp.dot(u, stage_k(u)) / 2
    elif stage_operator.kind == "controlled":
      u = flat_controls[stage_spec.param_name]
      x = stage_inputs[stage_spec.name]
      k_u, k_x = stage_k(u, x)
      cost += (jnp.dot(u, k_u) + tree_vdot(x, k_x)) / 2
    else:
      x = stage_inputs[stage_spec.name]
      k_x = stage_k(x)
      cost += tree_vdot(x, k_x) / 2

  return cost + _active_terminal_cost(final_state, final_q, final_lin_cost)


def _recover_control_gradients_from_active_execution_adjoint(controlled_stage_names, execution_stage_specs,
                                                             execution_stage_operators, stage_k_operators,
                                                             flat_controls, stage_inputs, final_state,
                                                             final_q, final_lin_cost):
  state_cotangent = _active_terminal_state_cotangent(final_state, final_q, final_lin_cost)
  recovered_by_layer = {}

  for stage_spec, stage_operator, stage_k in zip(reversed(execution_stage_specs),
                                                 reversed(execution_stage_operators),
                                                 reversed(stage_k_operators)):
    if stage_operator.kind == "passive":
      x = stage_inputs[stage_spec.name]
      stage_x_grad = stage_k(x)
      prev_state_cotangent = stage_operator.transpose(state_cotangent)
      state_cotangent = jax.tree_map(jnp.add, stage_x_grad, prev_state_cotangent)
      continue

    if stage_operator.kind == "control_only":
      recovered_by_layer[stage_spec.param_name] = stage_k(flat_controls[stage_spec.param_name]) + stage_operator.transpose(state_cotangent)
      break

    u = flat_controls[stage_spec.param_name]
    x = stage_inputs[stage_spec.name]
    stage_u_grad, stage_x_grad = stage_k(u, x)
    control_cotangent, prev_state_cotangent = stage_operator.transpose(state_cotangent)
    recovered_by_layer[stage_spec.param_name] = stage_u_grad + control_cotangent
    state_cotangent = jax.tree_map(jnp.add, stage_x_grad, prev_state_cotangent)

  return {layer_name: recovered_by_layer[layer_name] for layer_name in controlled_stage_names}


def _recover_preconditioner_gradients_from_active_execution_control_adjoint(block_structure, blocks, controlled_stage_names,
                                                                            execution_stage_specs, prepared_gradients, operators):
  execution_stage_operators, stage_k_operators, final_q, final_lin_cost = operators
  flat_controls = _materialize_active_flat_controls(
    block_structure, blocks, controlled_stage_names, prepared_gradients
  )
  stage_inputs, final_state = _rollout_active_execution_stages(
    execution_stage_specs, execution_stage_operators, flat_controls
  )
  control_gradients = _recover_control_gradients_from_active_execution_adjoint(
    controlled_stage_names, execution_stage_specs, execution_stage_operators, stage_k_operators,
    flat_controls, stage_inputs, final_state, final_q, final_lin_cost
  )

  recovered_by_layer = {
    layer_name: block_structure.preconditioner_param_pullback_flat_layer(
      blocks, layer_name, prepared_gradients[layer_name], control_gradients[layer_name]
    )
    for layer_name in controlled_stage_names
  }

  if isinstance(blocks, FrozenDict):
    return freeze(recovered_by_layer)
  return recovered_by_layer


def _active_execution_preconditioner_value_and_grad_from_control_adjoint(block_structure, blocks, controlled_stage_names,
                                                                         execution_stage_specs, prepared_gradients, operators):
  execution_stage_operators, stage_k_operators, final_q, final_lin_cost = operators
  flat_controls = _materialize_active_flat_controls(
    block_structure, blocks, controlled_stage_names, prepared_gradients
  )
  stage_inputs, final_state = _rollout_active_execution_stages(
    execution_stage_specs, execution_stage_operators, flat_controls
  )
  value = _active_execution_lqr_control_cost_from_rollout(
    execution_stage_specs, execution_stage_operators, stage_k_operators,
    flat_controls, stage_inputs, final_state, final_q, final_lin_cost
  )
  control_gradients = _recover_control_gradients_from_active_execution_adjoint(
    controlled_stage_names, execution_stage_specs, execution_stage_operators, stage_k_operators,
    flat_controls, stage_inputs, final_state, final_q, final_lin_cost
  )

  recovered_by_layer = {
    layer_name: block_structure.preconditioner_param_pullback_flat_layer(
      blocks, layer_name, prepared_gradients[layer_name], control_gradients[layer_name]
    )
    for layer_name in controlled_stage_names
  }

  if isinstance(blocks, FrozenDict):
    recovered_by_layer = freeze(recovered_by_layer)
  return value, recovered_by_layer

class BasePreconditioner(abc.ABC):
  def __init__(self,
               divergence_function,
               loss_fn, # As applied to the NN output
               block_structure,
               block_structure_init,
               model,
               network_params,
               optax_solver,
               trainstate_solver,
               preconditioner_update_steps,
               precond_rank,
               precond_identity_scaling,
               batch_solve_precond: bool = True,
               multibatch: bool = False,
               precond_on_update: bool =False,
               normalize_grad_for_lqr = True,
               warm_start_precond = True,
               damping: float = 0.0,
               allow_grad_inversion: bool = False,
               divergence_args_index = -1,
               llqr_operator_mode: str = "cached_exact",
               optax_solver_requires_value_and_grad: bool = False):
    self._divergence_function = divergence_function
    self._damping =damping
    self._loss_fn = loss_fn
    self._optax_solver = optax_solver
    self._trainstate_solver = trainstate_solver
    self._layer_names = list(network_params.keys())
    self._execution_stage_specs = tuple(getattr(model, "execution_stage_descriptors", ()))
    self._controlled_stage_specs = tuple(getattr(model, "controlled_stage_descriptors", ()))
    self._use_execution_stage_active_path = bool(getattr(model, "has_passive_stages", False))
    self._controlled_stage_unravel_fns = {}
    self._controlled_stage_flat_param_sizes = {}
    for layer_name in self._layer_names:
      flat_params, unravel_params_fn = ravel_pytree(network_params[layer_name])
      self._controlled_stage_unravel_fns[layer_name] = unravel_params_fn
      self._controlled_stage_flat_param_sizes[layer_name] = flat_params.size
    self._block_structure = BLOCK_STRUCTURE_DICT[block_structure](network_params, self._layer_names,
                                                                  block_structure_init, rank=precond_rank,
                                                                  identity_scale=precond_identity_scaling)
    self._block_structure_name = block_structure
    self._layer_modules = tuple(model.layers) if hasattr(model, "layers") else None
    # self._block_structure.make_blocks(network_params, model.layer_names)
    self._layer_apply = model.apply_block_from_params
    self._execution_stage_apply = getattr(model, "apply_block_from_params")
    self._model_apply = model.apply
    self._divergence_args_index = divergence_args_index
    self._preconditioner_update_steps = preconditioner_update_steps
    self._batch_solve_precond = batch_solve_precond
    self._multibatch = multibatch
    self._warm_start_precond = warm_start_precond
    self._optax_solver_requires_value_and_grad = optax_solver_requires_value_and_grad
    self._precond_on_update = precond_on_update
    self._allow_grad_inversion = allow_grad_inversion
    if llqr_operator_mode not in VALID_LLQR_OPERATOR_MODES:
      raise ValueError(
        f"Unknown llqr_operator_mode '{llqr_operator_mode}'. Expected one of {VALID_LLQR_OPERATOR_MODES}."
      )
    self._llqr_operator_mode = llqr_operator_mode
    if normalize_grad_for_lqr:
      self._normalize_grad_for_lqr_fn = normalize_gradient
    else:
      self._normalize_grad_for_lqr_fn = lambda _: _  # Nothing, identity fn

    self._update_preconditioner_fn = self._get_evaluate_lqr(self._optax_solver, self._preconditioner_update_steps,
                                                            batch_solve_precond=self._batch_solve_precond,
                                                            multibatch=self._multibatch,
                                                            precond_on_update=self._precond_on_update)
    # self._jit_apply_fn = jax.jit(
    #   lambda blocks, update: self._block_structure.matrix_product(blocks, update)
    # )
  def apply(self, blocks, update):
    # return self._jit_apply_fn(self._block_structure.blocks, update)
    return self._block_structure.matrix_product(blocks, update)

  def _should_snapshot_preconditioner_for_update(self, ema_decay):
    if ema_decay != 0:
      return True
    if hasattr(self._block_structure, "_memory"):
      return True
    return False

  def _should_use_shared_active_metadata_runtime(self):
    return self._use_execution_stage_active_path

  def _uses_lowmem_exact_k_mode(self):
    return self._llqr_operator_mode == "lowmem_exact_k"

  def _write_back_updated_preconditioner(self, updated_blocks, ema_decay, *, snapshot_preconditioner):
    if snapshot_preconditioner:
      self._block_structure.update_blocks(updated_blocks, ema_decay)
      return

    # Keep the original container identity on the no-snapshot branch.
    self._block_structure.blocks.update(updated_blocks)

  def get_stats(self):
    if hasattr(self._block_structure, "_memory"):
      precond_max, precond_min = pytree_max_min(self._block_structure.get_memory())
      precond_norm = pytree_l2_norm(self._block_structure.get_memory())
      per_layer_norm = get_per_layer_norm(self._block_structure.get_memory())
    else:
      precond_max, precond_min = pytree_max_min(self._block_structure.blocks)
      precond_norm = pytree_l2_norm(self._block_structure.blocks)
      per_layer_norm = get_per_layer_norm(self._block_structure.blocks)
    return precond_max, precond_min, precond_norm, per_layer_norm

  def get_precond_asymmetry(self):
    if hasattr(self._block_structure, "_memory"):
      skews = get_per_layer_skews(self._block_structure.get_memory())
    else:
      skews = get_per_layer_skews(self._block_structure.blocks)
    return skews


  def _get_evaluate_lqr(self, optax_solver=None, steps=1, batch_solve_precond=True, multibatch=False, precond_on_update=False):
    def compute_loss(_params, _other_model_variables, x, y):
      if type(_other_model_variables) is FrozenDict:
        _other_model_variables = dict(_other_model_variables)
      return self._loss_fn(self._model_apply({'params': _params}|_other_model_variables, x), y)

    if batch_solve_precond:
      def get_operators_and_gradients(params, other_model_variables, datapoint, trainstate_opt_state):
        inputs, targets = datapoint
        use_shared_active_metadata_runtime = self._should_use_shared_active_metadata_runtime()
        use_lowmem_exact_k_mode = self._uses_lowmem_exact_k_mode()
        if self._use_execution_stage_active_path:
          prepared_stage_metadata = prepare_active_execution_stage_metadata(
            params,
            self._execution_stage_specs,
            other_model_variables,
            param_unravel_fns=self._controlled_stage_unravel_fns,
            flat_param_sizes=self._controlled_stage_flat_param_sizes,
          ) if use_shared_active_metadata_runtime else None
          execution_stage_operators, states = lqr_active_execution_forward_operators_and_states(
            inputs, params, self._execution_stage_apply, self._execution_stage_specs, other_model_variables,
            prepared_stage_metadata=prepared_stage_metadata,
          )
        else:
          first_transition, first_transition_transpose, transitions, transition_transposes, states = (
            lqr_active_controllable_forward_operators_and_states(
            inputs, params, self._layer_apply, self._layer_names, other_model_variables)
          )
        if self._divergence_args_index is not None:
          div_arg = states[self._divergence_args_index]
        else:
          div_arg = None
        final_q, final_p, final_lin_cost = lqr_active_final_costs_and_adjoints(
          self._loss_fn, states[-1], targets, div_f=self._divergence_function, div_arg=div_arg
        )
        if self._use_execution_stage_active_path:
          if use_lowmem_exact_k_mode:
            stage_k_operators = lqr_active_execution_backward_hamiltonian_operators_lowmem(
              params, states, final_p, execution_stage_operators, self._execution_stage_apply,
              self._execution_stage_specs, self._damping, other_model_variables, layer_modules=self._layer_modules,
              prepared_stage_metadata=prepared_stage_metadata,
            )
          else:
            stage_k_operators = lqr_active_execution_backward_hamiltonian_operators(
              params, states, final_p, execution_stage_operators, self._execution_stage_apply,
              self._execution_stage_specs, self._damping, other_model_variables, layer_modules=self._layer_modules,
              prepared_stage_metadata=prepared_stage_metadata,
            )
          gradients = _recover_loss_gradients_from_execution_stage_transposes(
            self._layer_names, self._execution_stage_specs, execution_stage_operators, final_lin_cost,
            self._controlled_stage_unravel_fns, freeze_result=isinstance(params, FrozenDict)
          )
        else:
          if use_lowmem_exact_k_mode:
            first_k_backward, k_backward = lqr_active_controllable_backward_hamiltonian_operators_lowmem(
              params, states, final_p, transition_transposes, self._layer_apply, self._layer_names,
              self._damping, other_model_variables, layer_modules=self._layer_modules)
          else:
            first_k_backward, k_backward = lqr_active_controllable_backward_hamiltonian_operators(
              params, states, final_p, transition_transposes, self._layer_apply, self._layer_names,
              self._damping, other_model_variables, layer_modules=self._layer_modules)
          gradients = _recover_loss_gradients_from_transition_transposes(
            self._layer_names, transition_transposes, final_lin_cost,
            self._controlled_stage_unravel_fns,
            first_transition_transpose=first_transition_transpose,
            freeze_result=isinstance(params, FrozenDict))
        if precond_on_update:
          gradients, _ = self._trainstate_solver.update(gradients, trainstate_opt_state, params)
        gradients = self._normalize_grad_for_lqr_fn(gradients)
        gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient
        prepared_gradients = self._block_structure.prepare_train_vectors(gradients)

        if self._use_execution_stage_active_path:
          return prepared_gradients, (execution_stage_operators, stage_k_operators, final_q, final_lin_cost)

        return prepared_gradients, (
          first_transition, first_transition_transpose, transitions, transition_transposes,
          first_k_backward, k_backward, final_q, final_lin_cost)

      # def lqr_cost(_preconditioner, input_size, gradients, kernel_shapes, operators):
      def lqr_cost(_preconditioner, prepared_gradients, operators):
        if self._use_execution_stage_active_path:
          execution_stage_operators, stage_k_operators, final_q, final_lin_cost = operators
          flat_controls = _materialize_active_flat_controls(
            self._block_structure, _preconditioner, self._layer_names, prepared_gradients
          )
          stage_inputs, final_state = _rollout_active_execution_stages(
            self._execution_stage_specs, execution_stage_operators, flat_controls
          )
          return _active_execution_lqr_control_cost_from_rollout(
            self._execution_stage_specs, execution_stage_operators, stage_k_operators,
            flat_controls, stage_inputs, final_state, final_q, final_lin_cost
          )

        (first_transition, _, transitions, _,
         first_k_backward, k_backward, final_q, final_lin_cost) = operators
        cost = 0

        first_u = self._block_structure.train_matrix_product_flat_layer(
          _preconditioner, self._layer_names[0], prepared_gradients[self._layer_names[0]])
        cost += jnp.dot(first_u, first_k_backward(first_u)) / 2
        x = first_transition(first_u)

        for i, layer_name in enumerate(self._layer_names[1:]):
          u = self._block_structure.train_matrix_product_flat_layer(
            _preconditioner, layer_name, prepared_gradients[layer_name])
          k_u, k_x = k_backward[i](u, x)
          cost += (jnp.dot(u, k_u) + tree_vdot(x, k_x)) / 2
          x = transitions[i](u, x)

        cost += _active_terminal_cost(x, final_q, final_lin_cost)

        return cost

      def _sanitize_preconditioner_grads(grads):
        grads = jax.tree_map(Partial(jnp.nan_to_num, nan=0.0, posinf=1.0, neginf=-1.0), grads)
        return grads

      def lqr_grad_only_fn(_preconditioner, prepared_gradients, operators):
        if self._use_execution_stage_active_path:
          grads = _recover_preconditioner_gradients_from_active_execution_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, self._execution_stage_specs,
            prepared_gradients, operators
          )
        else:
          grads = _recover_preconditioner_gradients_from_active_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, prepared_gradients, operators)
        return _sanitize_preconditioner_grads(grads)

      def lqr_value_and_grad_fn(_preconditioner, prepared_gradients, operators):
        if self._use_execution_stage_active_path:
          value, grads = _active_execution_preconditioner_value_and_grad_from_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, self._execution_stage_specs,
            prepared_gradients, operators
          )
        else:
          value, grads = _active_preconditioner_value_and_grad_from_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, prepared_gradients, operators)
        grads = _sanitize_preconditioner_grads(grads)
        return value, grads

      if self._optax_solver_requires_value_and_grad:
        @Partial(jax.jit, donate_argnames=("preconditioner",))
        def _get_update(preconditioner, opt_state, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state):
          prepared_gradients, operators = get_operators_and_gradients(
            params, other_model_variables, datapoint, trainstate_opt_state)

          def update_step(carry, _):
            precond, opt_state = carry
            _, precond_grad = lqr_value_and_grad_fn(precond, prepared_gradients, operators)
            extra_args = {'value_and_grad_fn': Partial(
              lqr_value_and_grad_fn, prepared_gradients=prepared_gradients, operators=operators)}
            updates, opt_state = optax_solver.update(precond_grad, opt_state, precond, **extra_args)
            updates = jax.tree_map(lambda g: g * precond_lr, updates)
            new_precond = optax.apply_updates(precond, updates)
            return (new_precond, opt_state), None

          (final_precond, _), _ = jax.lax.scan(update_step,
                                               (preconditioner, opt_state),
                                               xs=None, length=steps, unroll=1)
          return jax.tree_map(jnp.nan_to_num, final_precond)
      else:
        @Partial(jax.jit, donate_argnames=("preconditioner",))
        def _get_update(preconditioner, opt_state, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state):
          prepared_gradients, operators = get_operators_and_gradients(
            params, other_model_variables, datapoint, trainstate_opt_state)

          def update_step(carry, _):
            precond, opt_state = carry
            precond_grad = lqr_grad_only_fn(precond, prepared_gradients, operators)
            updates, opt_state = optax_solver.update(precond_grad, opt_state, precond)
            updates = jax.tree_map(lambda g: g * precond_lr, updates)
            new_precond = optax.apply_updates(precond, updates)
            return (new_precond, opt_state), None

          (final_precond, _), _ = jax.lax.scan(update_step,
                                               (preconditioner, opt_state),
                                               xs=None, length=steps, unroll=1)
          return jax.tree_map(jnp.nan_to_num, final_precond)

      def get_update(preconditioner, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state,
                     snapshot_preconditioner=True):
        if snapshot_preconditioner:
          precond_input = _deep_copy_pytree(preconditioner)
        else:
          precond_input = preconditioner
        opt_state = optax_solver.init(precond_input)

        return _get_update(precond_input, opt_state, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state)

      return get_update
    else:
      @jax.jit
      def evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
        inputs, targets = datapoint
        transitions, transition_transposes, states = lqr_forward_matrices_and_states(
          inputs, params, self._layer_apply, self._layer_names, other_model_variables)
        if self._divergence_args_index is not None:
          div_arg = states[self._divergence_args_index]
        else:
          div_arg = None
        final_q, final_p, final_lin_cost = lqr_final_costs_and_adjoints(self._loss_fn, states[-1], targets,
                                                                        div_f=self._divergence_function,
                                                                        div_arg=div_arg)
        final_lin_cost = jnp.atleast_1d(final_lin_cost)
        q_backward, r_backward, m_backward, m_transpose_backward = lqr_backward_matrices_and_adjoints(params, states,
                                                                                                      final_p,
                                                                                                      transition_transposes,
                                                                                                      self._layer_apply,
                                                                                                      self._layer_names,
                                                                                                      self._damping,
                                                                                                      other_model_variables)
        gradients = jax.grad(compute_loss, argnums=0)(params, other_model_variables, inputs, targets)
        if precond_on_update:
          gradients, _ = self._trainstate_solver.update(gradients, trainstate_opt_state, params)
        gradients = self._normalize_grad_for_lqr_fn(gradients)
        gradients = jax.tree_map(lambda v: -1 * v, gradients)  # Starting update is negative gradient

        def lqr_cost(_preconditioner):
          cost = 0
          x = jnp.zeros(states[0].size)
          u_dict = self._block_structure.train_matrix_product(_preconditioner, gradients)
          for i, layer_name in enumerate(self._layer_names):
            u, _ = ravel_pytree(u_dict[layer_name])
            cost += (x.T @ q_backward[-i - 1](x) + u.T @ r_backward[-i - 1](u)) / 2 + u.T @ m_backward[-i - 1](x)
            x = transitions[i](u, x)

          # cost += x.T @ jnp.squeeze(final_lin_cost) + (x.T @ final_q(x)) / 2 # squeeze is causing problems
          x1 = jnp.ravel(x)  # () -> (1,), (n,1)/(1,n)/(n,) -> (n,)
          c1 = jnp.ravel(final_lin_cost)
          qx1 = jnp.ravel(final_q(x))  # assume final_q(x) ~ Qx

          cost += jnp.dot(x1, c1) + 0.5 * jnp.dot(x1, qx1)

          return cost

        local_precond_grad = jax.grad(lqr_cost, argnums=0)(preconditioner)
        return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), local_precond_grad)

      vmapped_evaluate_lqr_grad = jax.vmap(evaluate_lqr_grad, in_axes=(None, None, None, (0, 0), None))

      def get_precond_grad(preconditioner, params, other_model_variables, datapoint, trainstate_opt_state):
        # print(datapoint[0].shape)
        datapoint = tuple(jnp.expand_dims(x, axis=1) for x in datapoint)
        # print(datapoint[0].shape)
        grads = vmapped_evaluate_lqr_grad(preconditioner, params, other_model_variables, datapoint,
                                          trainstate_opt_state)
        return jax.tree_map(Partial(jnp.mean, axis=0), grads)

        # @timed_jit # switch back to jax.jit after debugging
        # # jax.jit
        # def get_update(preconditioner, params, other_model_variables, datapoint):
        #   opt_state = optax_solver.init(preconditioner)
        #   for _ in range(steps):
        #     precond_grad = get_precond_grad(preconditioner, params, other_model_variables, datapoint)
        #     _update, opt_state = optax_solver.update(precond_grad, opt_state)
        #     preconditioner = optax.apply_updates(preconditioner, _update)
        #   return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), preconditioner)
        #
        # return get_update

      if multibatch:
        def get_update(preconditioner, params, precond_lr, other_model_variables, dataloader, trainstate_opt_state):
          # Initialize the optimizer state for the preconditioner.
          opt_state = optax_solver.init(preconditioner)

          # Define a single update step to be run in the compiled loop.
          @jax.jit
          def update_step(precond, opt_state, datapoint):
            precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint, trainstate_opt_state)
            updates, opt_state = optax_solver.update(precond_grad, opt_state)
            updates = jax.tree_map(lambda g: g * precond_lr, updates)
            new_precond = optax.apply_updates(precond, updates)
            return (new_precond, opt_state)

          for _ in range(steps):
            precond, opt_state = update_step(preconditioner, opt_state, next(dataloader))
          # Safeguard against any numerical issues
          return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), precond)

      else:
        @jax.jit
        def get_update(preconditioner, params, precond_lr, other_model_variables, datapoint, trainstate_opt_state):
          # Initialize the optimizer state for the preconditioner.
          opt_state = optax_solver.init(preconditioner)

          # Define a single update step to be run in the compiled loop.
          def update_step(i, state):
            precond, opt_state = state
            precond_grad = get_precond_grad(precond, params, other_model_variables, datapoint, trainstate_opt_state)
            updates, opt_state = optax_solver.update(precond_grad, opt_state)
            updates = jax.tree_map(lambda g: g * precond_lr, updates)
            new_precond = optax.apply_updates(precond, updates)
            return (new_precond, opt_state)

          # Use a compiled loop to perform “steps” iterations.
          init_state = (preconditioner, opt_state)
          final_precond, _ = jax.lax.fori_loop(0, steps, update_step, init_state)
          # Safeguard against any numerical issues
          return jax.tree_map(Partial(jnp.nan_to_num, nan=1.0, posinf=1.0, neginf=1.0), final_precond)

      return get_update

  def update_preconditioner(self, params, dataloader, precond_lr, opt_state, precond_batch_size, ema_decay=0, other_model_variables=FrozenDict({})):
    """params is the current weights of the NN"""
    if self._multibatch:
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables, dataloader,
                                       opt_state), ema_decay)
    else:
      # Accumulate batches until we reach (or exceed) the requested preconditioner batch size
      # First batch to infer layout
      b0 = next(dataloader)
      layout = infer_batch_layout(b0)
      mode = layout["mode"]
      batch_axis = layout["batch_axis"]
      T = layout["T"]  # only used in text mode

      # Now start accumulation with the first batch already taken
      batches = [b0]
      # "Batch size" = size along batch_axis of x
      first_x = jax.tree_util.tree_leaves(b0)[0]
      if not isinstance(first_x, jnp.ndarray):
        first_x = jnp.asarray(first_x)
      current_B = first_x.shape[batch_axis]
      acc_size = int(current_B)

      while acc_size < precond_batch_size:
        b = next(dataloader)
        x_leaf = jax.tree_util.tree_leaves(b)[0]
        if not isinstance(x_leaf, jnp.ndarray):
          x_leaf = jnp.asarray(x_leaf)
        B = x_leaf.shape[batch_axis]
        batches.append(b)
        acc_size += int(B)

      # Concatenate according to layout
      def concat_fn(*xs):
        x0 = xs[0]
        if not isinstance(x0, jnp.ndarray):
          x0 = jnp.asarray(x0)

        if mode == "cv":
          # CV: batch axis always 0 for all arrays
          if x0.ndim >= 1:
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

        else:  # mode == "text"
          # For text:
          # - inputs: [T, B] -> concat along axis=1 (batch axis)
          # - targets: [T*B] -> concat along axis=0
          if x0.ndim >= 2:
            # assume it's something like [T, B, ...] and batch_axis is 1
            return jnp.concatenate(xs, axis=batch_axis)
          elif x0.ndim == 1:
            # flattened targets
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

      acc_batches = jax.tree_util.tree_map(concat_fn, *batches)

      # Clip to exactly precond_batch_size along batch dimension
      def clip_fn(x):
        if not isinstance(x, jnp.ndarray):
          x = jnp.asarray(x)

        if mode == "cv":
          # x: [B, ...] or [B]
          if x.ndim >= 1:
            return x[:precond_batch_size]
          else:
            return x

        else:  # mode == "text"
          if x.ndim >= 2:
            # inputs: [T, B_total] -> [T, precond_batch_size]
            # batch_axis is 1 here
            idx = [slice(None)] * x.ndim
            idx[batch_axis] = slice(0, precond_batch_size)
            return x[tuple(idx)]
          elif x.ndim == 1 and T is not None:
            # targets: [T*B_total] -> [T * precond_batch_size]
            return x[: T * precond_batch_size]
          else:
            return x

      acc_batches = jax.tree_util.tree_map(clip_fn, acc_batches)
      if self._warm_start_precond and not hasattr(self._block_structure, "_memory"):
        _blocks = self._block_structure.blocks
      else:
        _blocks = self._block_structure.reinit_blocks()
      should_snapshot = self._should_snapshot_preconditioner_for_update(ema_decay)
      updated_blocks = self._update_preconditioner_fn(
        _blocks,
        params,
        precond_lr,
        other_model_variables,
        acc_batches,
        opt_state,
        snapshot_preconditioner=should_snapshot,
      )
      if should_snapshot:
        self._write_back_updated_preconditioner(
          updated_blocks, ema_decay, snapshot_preconditioner=True
        )
      else:
        self._write_back_updated_preconditioner(
          updated_blocks, ema_decay, snapshot_preconditioner=False
        )

    if not self._allow_grad_inversion and self._block_structure_name in ('scalar', "diagonal"):
      # We clip to (almost) 0 those 2 structures to avoid gradient inversion
      self._block_structure.clip_blocks(min_for_block=1e-8)

    # print(self._block_structure.blocks["layers_2"])
    # print(self._block_structure.blocks)

  def compile_precond_updater(self, params, dataloader, precond_lr, opt_state, precond_batch_size, other_model_variables=FrozenDict({})):
    """For when we want to trigger jax compilation of _update_preconditioner_fn without applying the update"""
    if self._multibatch:
        blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables, dataloader,
                                       opt_state)
    else:
      # Accumulate batches until we reach (or exceed) the requested preconditioner batch size
      # First batch to infer layout
      b0 = next(dataloader)
      layout = infer_batch_layout(b0)
      mode = layout["mode"]
      batch_axis = layout["batch_axis"]
      T = layout["T"]  # only used in text mode

      # Now start accumulation with the first batch already taken
      batches = [b0]
      # "Batch size" = size along batch_axis of x
      first_x = jax.tree_util.tree_leaves(b0)[0]
      if not isinstance(first_x, jnp.ndarray):
        first_x = jnp.asarray(first_x)
      current_B = first_x.shape[batch_axis]
      acc_size = int(current_B)

      while acc_size < precond_batch_size:
        b = next(dataloader)
        x_leaf = jax.tree_util.tree_leaves(b)[0]
        if not isinstance(x_leaf, jnp.ndarray):
          x_leaf = jnp.asarray(x_leaf)
        B = x_leaf.shape[batch_axis]
        batches.append(b)
        acc_size += int(B)

      # Concatenate according to layout
      def concat_fn(*xs):
        x0 = xs[0]
        if not isinstance(x0, jnp.ndarray):
          x0 = jnp.asarray(x0)

        if mode == "cv":
          # CV: batch axis always 0 for all arrays
          if x0.ndim >= 1:
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

        else:  # mode == "text"
          # For text:
          # - inputs: [T, B] -> concat along axis=1 (batch axis)
          # - targets: [T*B] -> concat along axis=0
          if x0.ndim >= 2:
            # assume it's something like [T, B, ...] and batch_axis is 1
            return jnp.concatenate(xs, axis=batch_axis)
          elif x0.ndim == 1:
            # flattened targets
            return jnp.concatenate(xs, axis=0)
          else:
            return x0

      acc_batches = jax.tree_util.tree_map(concat_fn, *batches)

      # Clip to exactly precond_batch_size along batch dimension
      def clip_fn(x):
        if not isinstance(x, jnp.ndarray):
          x = jnp.asarray(x)

        if mode == "cv":
          # x: [B, ...] or [B]
          if x.ndim >= 1:
            return x[:precond_batch_size]
          else:
            return x

        else:  # mode == "text"
          if x.ndim >= 2:
            # inputs: [T, B_total] -> [T, precond_batch_size]
            # batch_axis is 1 here
            idx = [slice(None)] * x.ndim
            idx[batch_axis] = slice(0, precond_batch_size)
            return x[tuple(idx)]
          elif x.ndim == 1 and T is not None:
            # targets: [T*B_total] -> [T * precond_batch_size]
            return x[: T * precond_batch_size]
          else:
            return x

      acc_batches = jax.tree_util.tree_map(clip_fn, acc_batches)
      blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables,
                                     acc_batches, opt_state)

  def expose_blocks(self):
    return self._block_structure.blocks

  def load_blocks(self, saved_blocks):
    self._block_structure.update_blocks(saved_blocks)
