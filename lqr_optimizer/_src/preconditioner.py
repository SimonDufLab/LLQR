"""Preconditioner classes and their update rules."""
import abc
from functools import partial
import numbers

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
                             lqr_active_controllable_forward_operators_and_states_lowmem,
                             lqr_active_execution_forward_operators_and_states,
                             lqr_active_execution_forward_operators_and_states_lowmem,
                             lqr_final_costs_and_adjoints, lqr_active_final_costs_and_adjoints,
                             lqr_backward_matrices_and_adjoints,
                             lqr_backward_hamiltonian_operators,
                             lqr_active_controllable_backward_hamiltonian_operators,
                             lqr_active_controllable_backward_hamiltonian_operators_lowmem,
                             lqr_active_execution_backward_hamiltonian_operators,
                             lqr_active_execution_backward_hamiltonian_operators_lowmem,
                             prepare_active_execution_stage_metadata,
                             tree_vdot)
from lqr_optimizer._src.utils.build_lqr_segments import (
                             VALID_LQR_SEGMENT_SECOND_ORDER_MODES,
                             describe_lqr_segment_sample_separable_support,
                             lqr_active_segment_forward_operators_and_states,
                             lqr_active_segment_forward_operators_and_states_lowmem,
                             lqr_active_segment_backward_hamiltonian_operators,
                             lqr_active_segment_backward_hamiltonian_operators_lowmem)

BLOCK_STRUCTURE_DICT = {
  'dense': block_structures.DenseBlock,
  'diagonal': block_structures.DiagonalBlock,
  'scalar': block_structures.ScalarBlock,
  'kfac': block_structures.KroneckerBlock,
  'e-kfac': block_structures.EKFACBlock,
  'e-kfac-gpt': block_structures.GPTEKFACBlock,
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


VALID_LLQR_OPERATOR_MODES = ("cached_exact", "lowmem_exact_k", "lowmem_exact_full")
VALID_LLQR_CHECKPOINT_POLICIES = ("none", "dots_no_batch_dims")
VALID_LLQR_BATCH_UPDATE_MODES = ("full_batch", "chunked_lqr_segment")
VALID_LLQR_SECOND_ORDER_MODES = VALID_LQR_SEGMENT_SECOND_ORDER_MODES


def _tree_add(lhs, rhs):
  return jax.tree_map(jnp.add, lhs, rhs)


def _tree_scalar_mul(tree, scalar):
  return jax.tree_map(lambda leaf: leaf * scalar, tree)


def _batch_size_from_datapoint(datapoint, batch_axis):
  first_x = jax.tree_util.tree_leaves(datapoint)[0]
  if not isinstance(first_x, jnp.ndarray):
    first_x = jnp.asarray(first_x)
  return int(first_x.shape[batch_axis])


def _effective_loss_count(layout, batch_size):
  if layout["mode"] == "text":
    return int(layout["T"] * batch_size)
  return int(batch_size)


def _reshape_text_flat_targets(targets, layout, batch_size):
  if layout["mode"] != "text" or layout["T"] is None:
    raise ValueError("Text target reshaping requires a text layout with a known sequence length")
  return jnp.asarray(targets).reshape(int(layout["T"]), int(batch_size))


def _slice_batch_datapoint(datapoint, layout, start, size):
  mode = layout["mode"]
  batch_axis = layout["batch_axis"]
  batch_size = _batch_size_from_datapoint(datapoint, batch_axis)
  token_batch_size = None if layout["T"] is None else int(layout["T"] * batch_size)

  def slice_leaf(x):
    if not isinstance(x, jnp.ndarray):
      x = jnp.asarray(x)

    if mode == "cv":
      if x.ndim >= 1 and x.shape[0] == batch_size:
        return x[start:start + size]
      return x

    if x.ndim > batch_axis and x.shape[batch_axis] == batch_size:
      idx = [slice(None)] * x.ndim
      idx[batch_axis] = slice(start, start + size)
      return x[tuple(idx)]

    if x.ndim >= 1 and x.shape[0] == batch_size:
      return x[start:start + size]

    if token_batch_size is not None and x.ndim >= 1 and x.shape[0] == token_batch_size:
      targets_time_batch = _reshape_text_flat_targets(x, layout, batch_size)
      return jnp.ravel(targets_time_batch[:, start:start + size])

    return x

  return jax.tree_util.tree_map(slice_leaf, datapoint)


def _concat_preconditioner_batches(batches, layout, precond_batch_size):
  mode = layout["mode"]
  batch_axis = layout["batch_axis"]

  def concat_fn(*xs):
    x0 = xs[0]
    if not isinstance(x0, jnp.ndarray):
      x0 = jnp.asarray(x0)

    if mode == "cv":
      if x0.ndim >= 1:
        return jnp.concatenate(xs, axis=0)
      return x0

    if (
      layout["T"] is not None
      and x0.ndim == 1
      and all(jnp.asarray(x).ndim == 1 and jnp.asarray(x).shape[0] % layout["T"] == 0 for x in xs)
    ):
      reshaped = [_reshape_text_flat_targets(x, layout, jnp.asarray(x).shape[0] // layout["T"]) for x in xs]
      return jnp.ravel(jnp.concatenate(reshaped, axis=1))

    if x0.ndim > batch_axis:
      return jnp.concatenate(xs, axis=batch_axis)
    if x0.ndim == 1:
      return jnp.concatenate(xs, axis=0)
    return x0

  accumulated = jax.tree_util.tree_map(concat_fn, *batches)
  return _slice_batch_datapoint(accumulated, layout, 0, precond_batch_size)


def _take_full_preconditioner_datapoint(dataloader, precond_batch_size):
  first_batch = next(dataloader)
  layout = infer_batch_layout(first_batch)
  batch_axis = layout["batch_axis"]

  batches = [first_batch]
  accumulated_size = _batch_size_from_datapoint(first_batch, batch_axis)
  while accumulated_size < precond_batch_size:
    batch = next(dataloader)
    batches.append(batch)
    accumulated_size += _batch_size_from_datapoint(batch, batch_axis)

  return layout, _concat_preconditioner_batches(batches, layout, precond_batch_size)


def _take_preconditioner_chunk_datapoints(dataloader, precond_batch_size, batch_chunk_size):
  if batch_chunk_size is None:
    raise ValueError("batch_chunk_size must be set for chunked preconditioner ingestion")

  current_batch = next(dataloader)
  layout = infer_batch_layout(current_batch)
  batch_axis = layout["batch_axis"]
  current_batch_size = _batch_size_from_datapoint(current_batch, batch_axis)
  current_offset = 0
  accumulated_size = 0
  chunk_datapoints = []
  chunk_weights = []

  while accumulated_size < precond_batch_size:
    available = current_batch_size - current_offset
    remaining_total = precond_batch_size - accumulated_size
    take_size = min(batch_chunk_size, available, remaining_total)
    chunk_datapoints.append(_slice_batch_datapoint(current_batch, layout, current_offset, take_size))
    chunk_weights.append(_effective_loss_count(layout, take_size))
    accumulated_size += take_size
    current_offset += take_size

    if current_offset == current_batch_size and accumulated_size < precond_batch_size:
      current_batch = next(dataloader)
      current_batch_size = _batch_size_from_datapoint(current_batch, batch_axis)
      current_offset = 0

  return layout, chunk_datapoints, chunk_weights


def _supports_chunked_execution_stage_update_layout(layout):
  return layout["mode"] in ("cv", "text")


def _normalize_execution_stage_update_datapoint(datapoint, layout):
  if layout["mode"] != "text":
    return layout, datapoint

  inputs, targets = datapoint
  inputs = jnp.asarray(inputs)
  batch_size = int(inputs.shape[layout["batch_axis"]])
  targets_time_batch = _reshape_text_flat_targets(targets, layout, batch_size)
  normalized_inputs = jnp.swapaxes(inputs, 0, layout["batch_axis"])
  normalized_targets = jnp.ravel(jnp.swapaxes(targets_time_batch, 0, 1))
  normalized_layout = {
    "mode": "cv",
    "batch_axis": 0,
    "T": layout["T"],
  }
  return normalized_layout, (normalized_inputs, normalized_targets)


def _normalize_execution_stage_update_chunks(chunk_datapoints, layout):
  normalized_layout = layout
  normalized_datapoints = []
  for chunk_datapoint in chunk_datapoints:
    normalized_layout, normalized_datapoint = _normalize_execution_stage_update_datapoint(
      chunk_datapoint,
      layout,
    )
    normalized_datapoints.append(normalized_datapoint)
  return normalized_layout, normalized_datapoints


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


def _recover_loss_gradients_from_lqr_segment_transposes(params_or_controlled_stage_names,
                                                        controlled_stage_names_or_lqr_segment_specs,
                                                        lqr_segment_specs_or_segment_operators,
                                                        segment_operators_or_final_lin_cost,
                                                        final_lin_cost_or_unravel_params_fns,
                                                        unravel_params_fns=None,
                                                        freeze_result=False):
  if isinstance(params_or_controlled_stage_names, (dict, FrozenDict)):
    params = params_or_controlled_stage_names
    controlled_stage_names = controlled_stage_names_or_lqr_segment_specs
    lqr_segment_specs = lqr_segment_specs_or_segment_operators
    segment_operators = segment_operators_or_final_lin_cost
    final_lin_cost = final_lin_cost_or_unravel_params_fns
    if unravel_params_fns is None:
      unravel_params_fns = {
        layer_name: ravel_pytree(params[layer_name])[1]
        for layer_name in controlled_stage_names
      }
    freeze_result = isinstance(params, FrozenDict)
  else:
    controlled_stage_names = params_or_controlled_stage_names
    lqr_segment_specs = controlled_stage_names_or_lqr_segment_specs
    segment_operators = lqr_segment_specs_or_segment_operators
    final_lin_cost = segment_operators_or_final_lin_cost
    unravel_params_fns = final_lin_cost_or_unravel_params_fns

  state_cotangent = final_lin_cost
  recovered_by_layer = {}

  for segment_spec, segment_operator in zip(reversed(lqr_segment_specs), reversed(segment_operators)):
    if segment_operator.kind == "passive":
      state_cotangent = segment_operator.transpose(state_cotangent)
      continue

    if segment_operator.kind == "control_only":
      param_cotangents = segment_operator.transpose(state_cotangent)
      for param_name in segment_spec.controlled_param_names:
        unravel_params_fn = unravel_params_fns[param_name]
        recovered_by_layer[param_name] = unravel_params_fn(
          jnp.ravel(jnp.atleast_1d(param_cotangents[param_name]))
        )
      break

    param_cotangents, state_cotangent = segment_operator.transpose(state_cotangent)
    for param_name in segment_spec.controlled_param_names:
      unravel_params_fn = unravel_params_fns[param_name]
      recovered_by_layer[param_name] = unravel_params_fn(
        jnp.ravel(jnp.atleast_1d(param_cotangents[param_name]))
      )

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


def _segment_flat_controls(flat_controls, segment_spec):
  return {
    param_name: flat_controls[param_name]
    for param_name in segment_spec.controlled_param_names
  }


def _store_segment_control_gradients(recovered_by_layer, segment_spec, segment_gradients):
  for param_name in segment_spec.controlled_param_names:
    recovered_by_layer[param_name] = segment_gradients[param_name]


def _add_segment_control_gradients(lhs, rhs):
  return {name: lhs[name] + rhs[name] for name in lhs}


def _rollout_active_lqr_segments(lqr_segment_specs, segment_operators, flat_controls):
  segment_inputs = {}
  x = None
  for segment_spec, segment_operator in zip(lqr_segment_specs, segment_operators):
    if segment_operator.kind == "control_only":
      segment_inputs[segment_spec.name] = None
      x = segment_operator.forward(_segment_flat_controls(flat_controls, segment_spec))
    elif segment_operator.kind == "controlled":
      if x is None:
        raise ValueError("Encountered a controlled LLQR segment before the first control-only segment.")
      segment_inputs[segment_spec.name] = x
      x = segment_operator.forward(_segment_flat_controls(flat_controls, segment_spec), x)
    else:
      if x is None:
        raise ValueError("Passive LLQR segments before the first controlled segment are not supported.")
      segment_inputs[segment_spec.name] = x
      x = segment_operator.forward(x)
  return segment_inputs, x


def _active_lqr_segment_control_cost_from_rollout(lqr_segment_specs, segment_operators, segment_k_operators,
                                                  flat_controls, segment_inputs, final_state,
                                                  final_q, final_lin_cost):
  cost = 0.0
  for segment_spec, segment_operator, segment_k in zip(lqr_segment_specs, segment_operators, segment_k_operators):
    if segment_operator.kind == "control_only":
      u = _segment_flat_controls(flat_controls, segment_spec)
      cost += tree_vdot(u, segment_k(u)) / 2
    elif segment_operator.kind == "controlled":
      u = _segment_flat_controls(flat_controls, segment_spec)
      x = segment_inputs[segment_spec.name]
      k_u, k_x = segment_k(u, x)
      cost += (tree_vdot(u, k_u) + tree_vdot(x, k_x)) / 2
    else:
      x = segment_inputs[segment_spec.name]
      k_x = segment_k(x)
      cost += tree_vdot(x, k_x) / 2

  return cost + _active_terminal_cost(final_state, final_q, final_lin_cost)


def _recover_control_gradients_from_active_lqr_segment_adjoint(controlled_stage_names, lqr_segment_specs,
                                                               segment_operators, segment_k_operators,
                                                               flat_controls, segment_inputs, final_state,
                                                               final_q, final_lin_cost):
  state_cotangent = _active_terminal_state_cotangent(final_state, final_q, final_lin_cost)
  recovered_by_layer = {}

  for segment_spec, segment_operator, segment_k in zip(reversed(lqr_segment_specs),
                                                       reversed(segment_operators),
                                                       reversed(segment_k_operators)):
    if segment_operator.kind == "passive":
      x = segment_inputs[segment_spec.name]
      stage_x_grad = segment_k(x)
      prev_state_cotangent = segment_operator.transpose(state_cotangent)
      state_cotangent = jax.tree_map(jnp.add, stage_x_grad, prev_state_cotangent)
      continue

    if segment_operator.kind == "control_only":
      u = _segment_flat_controls(flat_controls, segment_spec)
      segment_gradients = _add_segment_control_gradients(segment_k(u), segment_operator.transpose(state_cotangent))
      _store_segment_control_gradients(recovered_by_layer, segment_spec, segment_gradients)
      break

    u = _segment_flat_controls(flat_controls, segment_spec)
    x = segment_inputs[segment_spec.name]
    stage_u_grad, stage_x_grad = segment_k(u, x)
    control_cotangent, prev_state_cotangent = segment_operator.transpose(state_cotangent)
    _store_segment_control_gradients(
      recovered_by_layer,
      segment_spec,
      _add_segment_control_gradients(stage_u_grad, control_cotangent),
    )
    state_cotangent = jax.tree_map(jnp.add, stage_x_grad, prev_state_cotangent)

  return {layer_name: recovered_by_layer[layer_name] for layer_name in controlled_stage_names}


def _recover_preconditioner_gradients_from_active_lqr_segment_control_adjoint(block_structure, blocks,
                                                                              controlled_stage_names,
                                                                              lqr_segment_specs,
                                                                              prepared_gradients,
                                                                              operators):
  segment_operators, segment_k_operators, final_q, final_lin_cost = operators
  flat_controls = _materialize_active_flat_controls(
    block_structure, blocks, controlled_stage_names, prepared_gradients
  )
  segment_inputs, final_state = _rollout_active_lqr_segments(
    lqr_segment_specs, segment_operators, flat_controls
  )
  control_gradients = _recover_control_gradients_from_active_lqr_segment_adjoint(
    controlled_stage_names, lqr_segment_specs, segment_operators, segment_k_operators,
    flat_controls, segment_inputs, final_state, final_q, final_lin_cost
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


def _active_lqr_segment_preconditioner_value_and_grad_from_control_adjoint(block_structure, blocks,
                                                                           controlled_stage_names,
                                                                           lqr_segment_specs,
                                                                           prepared_gradients,
                                                                           operators):
  segment_operators, segment_k_operators, final_q, final_lin_cost = operators
  flat_controls = _materialize_active_flat_controls(
    block_structure, blocks, controlled_stage_names, prepared_gradients
  )
  segment_inputs, final_state = _rollout_active_lqr_segments(
    lqr_segment_specs, segment_operators, flat_controls
  )
  value = _active_lqr_segment_control_cost_from_rollout(
    lqr_segment_specs, segment_operators, segment_k_operators,
    flat_controls, segment_inputs, final_state, final_q, final_lin_cost
  )
  control_gradients = _recover_control_gradients_from_active_lqr_segment_adjoint(
    controlled_stage_names, lqr_segment_specs, segment_operators, segment_k_operators,
    flat_controls, segment_inputs, final_state, final_q, final_lin_cost
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
               llqr_checkpoint_policy: str = "none",
               llqr_use_fast_paths: bool = True,
               llqr_batch_update_mode: str = "full_batch",
               llqr_batch_update_chunk_size = None,
               llqr_second_order_mode: str = "batched_exact",
               llqr_second_order_chunk_size = None,
               use_spectral_norm: bool = False,
               optax_solver_requires_value_and_grad: bool = False):
    self._divergence_function = divergence_function
    self._damping =damping
    self._loss_fn = loss_fn
    self._optax_solver = optax_solver
    self._trainstate_solver = trainstate_solver
    self._layer_names = list(network_params.keys())
    self._execution_stage_specs = tuple(getattr(model, "execution_stage_descriptors", ()))
    self._controlled_stage_specs = tuple(getattr(model, "controlled_stage_descriptors", ()))
    self._lqr_segment_specs = tuple(getattr(model, "resolved_lqr_segment_descriptors", ()))
    self._use_execution_stage_active_path = bool(getattr(model, "has_passive_stages", False))
    if self._use_execution_stage_active_path and self._lqr_segment_specs and divergence_args_index not in (None, -1):
      raise ValueError(
        "Grouped LLQR segments currently support terminal divergence_args_index only."
      )
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
    if llqr_checkpoint_policy not in VALID_LLQR_CHECKPOINT_POLICIES:
      raise ValueError(
        f"Unknown llqr_checkpoint_policy '{llqr_checkpoint_policy}'. Expected one of {VALID_LLQR_CHECKPOINT_POLICIES}."
      )
    if llqr_batch_update_mode not in VALID_LLQR_BATCH_UPDATE_MODES:
      raise ValueError(
        "Unknown llqr_batch_update_mode "
        f"'{llqr_batch_update_mode}'. Expected one of {VALID_LLQR_BATCH_UPDATE_MODES}."
      )
    if llqr_second_order_mode not in VALID_LLQR_SECOND_ORDER_MODES:
      raise ValueError(
        "Unknown llqr_second_order_mode "
        f"'{llqr_second_order_mode}'. Expected one of {VALID_LLQR_SECOND_ORDER_MODES}."
      )
    if llqr_second_order_mode == "batched_exact":
      if llqr_second_order_chunk_size is not None:
        raise ValueError("llqr_second_order_chunk_size must be null when llqr_second_order_mode='batched_exact'.")
    else:
      if llqr_second_order_chunk_size is not None:
        if (
            not isinstance(llqr_second_order_chunk_size, numbers.Integral)
            or isinstance(llqr_second_order_chunk_size, bool)
            or int(llqr_second_order_chunk_size) <= 0
        ):
          raise ValueError(
            "llqr_second_order_chunk_size must be a positive integer when "
            "llqr_second_order_mode='sample_separable_exact'."
          )
      if not batch_solve_precond:
        raise ValueError("llqr_second_order_mode='sample_separable_exact' requires batch_solve_precond=True.")
      if not self._uses_grouped_execution_stage_operator_path():
        raise ValueError(
          "llqr_second_order_mode='sample_separable_exact' requires a grouped LLQR segment operator path."
        )
      unsupported_segments = self._sample_separable_segment_support_details()[
        "sample_separable_unsupported_segments"
      ]
      if unsupported_segments:
        unsupported = ", ".join(
          f"{item['name']}:{item['reason']}" for item in unsupported_segments
        )
        raise ValueError(
          "llqr_second_order_mode='sample_separable_exact' requires sample-separable "
          f"LLQR segment metadata; unsupported segments: {unsupported}"
        )
    if llqr_batch_update_mode == "full_batch":
      if llqr_batch_update_chunk_size is not None:
        raise ValueError("llqr_batch_update_chunk_size must be null when llqr_batch_update_mode='full_batch'.")
    else:
      if not batch_solve_precond:
        raise ValueError("llqr_batch_update_mode='chunked_lqr_segment' requires batch_solve_precond=True.")
      if llqr_operator_mode != "cached_exact":
        raise ValueError("llqr_batch_update_mode='chunked_lqr_segment' requires llqr_operator_mode='cached_exact'.")
      if not isinstance(llqr_batch_update_chunk_size, numbers.Integral) or bool(llqr_batch_update_chunk_size) is False:
        raise ValueError(
          "llqr_batch_update_chunk_size must be a positive integer when llqr_batch_update_mode='chunked_lqr_segment'."
        )
      if int(llqr_batch_update_chunk_size) <= 0:
        raise ValueError(
          "llqr_batch_update_chunk_size must be a positive integer when llqr_batch_update_mode='chunked_lqr_segment'."
        )
    self._llqr_operator_mode = llqr_operator_mode
    self._llqr_checkpoint_policy = llqr_checkpoint_policy
    self._llqr_use_fast_paths = bool(llqr_use_fast_paths)
    self._use_spectral_norm = bool(use_spectral_norm)
    self._llqr_batch_update_mode = llqr_batch_update_mode
    self._llqr_batch_update_chunk_size = None if llqr_batch_update_chunk_size is None else int(llqr_batch_update_chunk_size)
    self._llqr_second_order_mode = llqr_second_order_mode
    self._llqr_second_order_chunk_size = (
      None if llqr_second_order_chunk_size is None else int(llqr_second_order_chunk_size)
    )
    if (
      self._llqr_batch_update_mode == "chunked_lqr_segment"
      and self._use_execution_stage_active_path
      and not self._lqr_segment_specs
    ):
      raise ValueError(
        "Chunked grouped LLQR updates require resolved LLQR segment descriptors."
      )
    if normalize_grad_for_lqr:
      self._normalize_grad_for_lqr_fn = normalize_gradient
    else:
      self._normalize_grad_for_lqr_fn = lambda _: _  # Nothing, identity fn

    self._update_preconditioner_fn = self._get_evaluate_lqr(self._optax_solver, self._preconditioner_update_steps,
                                                            batch_solve_precond=self._batch_solve_precond,
                                                            multibatch=self._multibatch,
                                                            precond_on_update=self._precond_on_update)
    self._llqr_batch_update_gate = self._llqr_batch_update_gate_details()
    self._last_llqr_batch_update_route = None
    self._grouped_chunked_update_loss_gradients_fn = None
    self._grouped_chunked_update_grad_only_fn = None
    self._grouped_chunked_update_value_and_grad_fn = None
    if self._uses_chunked_lqr_segment_update_path():
      (self._grouped_chunked_update_loss_gradients_fn,
       self._grouped_chunked_update_grad_only_fn,
       self._grouped_chunked_update_value_and_grad_fn) = self._get_grouped_chunked_update_helpers(
         precond_on_update=self._precond_on_update
       )
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

  def _uses_lowmem_exact_full_mode(self):
    return self._llqr_operator_mode == "lowmem_exact_full"

  def _uses_lowmem_active_k_mode(self):
    return self._llqr_operator_mode in ("lowmem_exact_k", "lowmem_exact_full")

  def _uses_lowmem_active_transition_mode(self):
    return self._llqr_operator_mode == "lowmem_exact_full"

  def _resolved_llqr_checkpoint_policy(self):
    if not self._uses_lowmem_exact_full_mode():
      return "none"
    return self._llqr_checkpoint_policy

  def _uses_active_fast_paths(self):
    return self._llqr_use_fast_paths and not self._use_spectral_norm

  def _uses_grouped_execution_stage_operator_path(self):
    return self._use_execution_stage_active_path and bool(self._lqr_segment_specs)

  def _uses_grouped_full_batch_execution_stage_path(self):
    return (
      self._uses_grouped_execution_stage_operator_path()
      and self._resolved_llqr_batch_update_mode() == "full_batch"
    )

  def _uses_grouped_chunked_execution_stage_path(self):
    return (
      self._uses_chunked_lqr_segment_update_path()
      and self._uses_grouped_execution_stage_operator_path()
    )

  def describe_llqr_operator_route(self):
    if self._uses_grouped_execution_stage_operator_path():
      operator_granularity = "lqr_segment"
    elif self._use_execution_stage_active_path:
      operator_granularity = "execution_stage"
    else:
      operator_granularity = "controlled_layer"
    route = {
      "operator_granularity": operator_granularity,
      "execution_stage_count": len(self._execution_stage_specs),
      "lqr_segment_count": len(self._lqr_segment_specs),
      "controlled_param_count": len(self._layer_names),
      "llqr_operator_mode": self._llqr_operator_mode,
      "uses_grouped_full_batch_path": self._uses_grouped_full_batch_execution_stage_path(),
      "uses_grouped_chunked_path": self._uses_grouped_chunked_execution_stage_path(),
      "uses_fast_paths": self._uses_active_fast_paths(),
    }
    route.update(self._second_order_route_details())
    return route

  def _uses_chunked_lqr_segment_update_mode(self):
    return self._llqr_batch_update_mode == "chunked_lqr_segment"

  def _resolved_llqr_batch_update_mode(self):
    return self._llqr_batch_update_mode

  def _resolved_llqr_batch_update_chunk_size(self):
    return self._llqr_batch_update_chunk_size

  def _uses_sample_separable_second_order(self):
    return self._llqr_second_order_mode == "sample_separable_exact"

  def _sample_separable_segment_support_details(self):
    if not self._lqr_segment_specs:
      return {
        "sample_separable_supported_segment_count": 0,
        "sample_separable_unsupported_segments": [],
      }
    return describe_lqr_segment_sample_separable_support(self._lqr_segment_specs)

  def _resolved_second_order_chunk_size(self, *, fallback_batch_size):
    if not self._uses_sample_separable_second_order():
      return None
    if self._llqr_second_order_chunk_size is not None:
      return self._llqr_second_order_chunk_size
    return int(fallback_batch_size)

  def _second_order_route_details(self, *, precond_batch_size=None, batch_axis=None,
                                  chunk_datapoints=None, fallback_chunk_size=None):
    details = {
      "second_order_mode": self._llqr_second_order_mode,
      "configured_second_order_chunk_size": self._llqr_second_order_chunk_size,
      "uses_sample_separable_second_order": self._uses_sample_separable_second_order(),
      "resolved_second_order_chunk_size": None,
      "second_order_batch_axis": None,
      "second_order_chunk_count": None,
    }
    details.update(self._sample_separable_segment_support_details())
    if not self._uses_sample_separable_second_order():
      return details

    resolved_batch_axis = 0 if batch_axis is None else int(batch_axis)
    if fallback_chunk_size is None:
      fallback_chunk_size = precond_batch_size
    if fallback_chunk_size is None:
      if self._llqr_second_order_chunk_size is not None:
        details.update({
          "resolved_second_order_chunk_size": int(self._llqr_second_order_chunk_size),
          "second_order_batch_axis": resolved_batch_axis,
        })
      return details
    resolved_chunk_size = self._resolved_second_order_chunk_size(
      fallback_batch_size=fallback_chunk_size
    )

    if chunk_datapoints is None:
      if precond_batch_size is None:
        chunk_count = None
      else:
        chunk_count = (int(precond_batch_size) + resolved_chunk_size - 1) // resolved_chunk_size
    else:
      chunk_count = sum(
        (_batch_size_from_datapoint(chunk_datapoint, resolved_batch_axis) + resolved_chunk_size - 1)
        // resolved_chunk_size
        for chunk_datapoint in chunk_datapoints
      )

    details.update({
      "resolved_second_order_chunk_size": int(resolved_chunk_size),
      "second_order_batch_axis": resolved_batch_axis,
      "second_order_chunk_count": None if chunk_count is None else int(chunk_count),
    })
    return details

  def _llqr_batch_update_gate_details(self):
    blocked_by = []
    if not self._uses_chunked_lqr_segment_update_mode():
      blocked_by.append("mode_not_chunked_lqr_segment")
    if not self._use_execution_stage_active_path:
      blocked_by.append("no_execution_stage_active_path")
    if not self._batch_solve_precond:
      blocked_by.append("batch_solve_precond_disabled")
    if self._multibatch:
      blocked_by.append("multibatch_enabled")
    gate_details = {
      "batch_update_mode": self._resolved_llqr_batch_update_mode(),
      "batch_update_chunk_size": self._resolved_llqr_batch_update_chunk_size(),
      "use_execution_stage_active_path": bool(self._use_execution_stage_active_path),
      "batch_solve_precond": bool(self._batch_solve_precond),
      "multibatch": bool(self._multibatch),
      "uses_streamed_execution_stage_update_path": not blocked_by,
      "uses_streamed_lqr_segment_update_path": (
        not blocked_by and self._uses_grouped_execution_stage_operator_path()
      ),
      "blocked_by": blocked_by,
    }
    gate_details.update(self._second_order_route_details())
    return gate_details

  def describe_llqr_batch_update_gate(self):
    return dict(self._llqr_batch_update_gate)

  def _serialize_batch_layout(self, layout):
    if layout is None:
      return None
    serialized = {}
    for key in ("mode", "batch_axis", "T"):
      value = layout.get(key)
      if value is None:
        serialized[key] = None
      elif isinstance(value, numbers.Integral):
        serialized[key] = int(value)
      else:
        serialized[key] = value
    return serialized

  def _record_llqr_batch_update_route(self, *, operation, route, precond_batch_size,
                                              layout=None, normalized_layout=None,
                                              chunk_datapoints=None, chunk_weights=None,
                                              fallback_reason=None):
    route_info = self.describe_llqr_batch_update_gate()
    route_info.update({
      "operation": operation,
      "route": route,
      "precond_batch_size": int(precond_batch_size),
    })
    route_info.update(self.describe_llqr_operator_route())
    route_batch_axis = None
    if normalized_layout is not None:
      route_batch_axis = normalized_layout.get("batch_axis")
    elif layout is not None:
      route_batch_axis = layout.get("batch_axis")
    fallback_chunk_size = precond_batch_size
    if chunk_datapoints is not None and self._llqr_second_order_chunk_size is None:
      resolved_batch_axis = 0 if route_batch_axis is None else int(route_batch_axis)
      fallback_chunk_size = max(
        _batch_size_from_datapoint(chunk_datapoint, resolved_batch_axis)
        for chunk_datapoint in chunk_datapoints
      )
    elif chunk_datapoints is not None:
      fallback_chunk_size = self._resolved_llqr_batch_update_chunk_size()
    route_info.update(self._second_order_route_details(
      precond_batch_size=precond_batch_size,
      batch_axis=route_batch_axis,
      chunk_datapoints=chunk_datapoints,
      fallback_chunk_size=fallback_chunk_size,
    ))
    if layout is not None:
      route_info["layout"] = self._serialize_batch_layout(layout)
      route_info["layout_supported"] = bool(_supports_chunked_execution_stage_update_layout(layout))
    if normalized_layout is not None:
      route_info["normalized_layout"] = self._serialize_batch_layout(normalized_layout)
    if chunk_datapoints is not None:
      route_info["chunk_count"] = len(chunk_datapoints)
    if chunk_weights is not None:
      route_info["chunk_weights"] = [int(weight) for weight in chunk_weights]
    if fallback_reason is not None:
      route_info["fallback_reason"] = fallback_reason
    self._last_llqr_batch_update_route = route_info
    return route_info

  def describe_last_llqr_batch_update_route(self):
    if self._last_llqr_batch_update_route is None:
      return None
    return dict(self._last_llqr_batch_update_route)

  def _uses_chunked_lqr_segment_update_path(self):
    return (
      self._uses_chunked_lqr_segment_update_mode()
      and self._use_execution_stage_active_path
      and self._batch_solve_precond
      and not self._multibatch
    )

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
    use_grouped_execution_stage_operator_path = self._uses_grouped_execution_stage_operator_path()

    def compute_loss(_params, _other_model_variables, x, y):
      if type(_other_model_variables) is FrozenDict:
        _other_model_variables = dict(_other_model_variables)
      return self._loss_fn(self._model_apply({'params': _params}|_other_model_variables, x), y)

    if batch_solve_precond:
      def get_operators_and_gradients(params, other_model_variables, datapoint, trainstate_opt_state):
        inputs, targets = datapoint
        second_order_batch_axis = 0 if self._uses_sample_separable_second_order() else None
        second_order_chunk_size = None
        if self._uses_sample_separable_second_order():
          second_order_chunk_size = self._resolved_second_order_chunk_size(
            fallback_batch_size=_batch_size_from_datapoint(datapoint, second_order_batch_axis)
          )
        use_shared_active_metadata_runtime = self._should_use_shared_active_metadata_runtime()
        use_lowmem_active_transition_mode = self._uses_lowmem_active_transition_mode()
        use_lowmem_active_k_mode = self._uses_lowmem_active_k_mode()
        resolved_checkpoint_policy = self._resolved_llqr_checkpoint_policy()
        batch_axis = None
        if self._uses_chunked_lqr_segment_update_mode():
          batch_axis = infer_batch_layout(datapoint)["batch_axis"]
        if self._use_execution_stage_active_path:
          prepared_stage_metadata = prepare_active_execution_stage_metadata(
            params,
            self._execution_stage_specs,
            other_model_variables,
            param_unravel_fns=self._controlled_stage_unravel_fns,
            flat_param_sizes=self._controlled_stage_flat_param_sizes,
          ) if use_shared_active_metadata_runtime else None
          if use_grouped_execution_stage_operator_path:
            if use_lowmem_active_transition_mode:
              segment_operators, states = lqr_active_segment_forward_operators_and_states_lowmem(
                inputs, params, self._execution_stage_apply, self._lqr_segment_specs, other_model_variables,
                prepared_stage_metadata=prepared_stage_metadata,
                checkpoint_policy=resolved_checkpoint_policy,
              )
            else:
              segment_operators, states = lqr_active_segment_forward_operators_and_states(
                inputs, params, self._execution_stage_apply, self._lqr_segment_specs, other_model_variables,
                prepared_stage_metadata=prepared_stage_metadata,
              )
          elif use_lowmem_active_transition_mode:
            execution_stage_operators, states = lqr_active_execution_forward_operators_and_states_lowmem(
              inputs, params, self._execution_stage_apply, self._execution_stage_specs, other_model_variables,
              prepared_stage_metadata=prepared_stage_metadata,
              checkpoint_policy=resolved_checkpoint_policy,
            )
          else:
            execution_stage_operators, states = lqr_active_execution_forward_operators_and_states(
              inputs, params, self._execution_stage_apply, self._execution_stage_specs, other_model_variables,
              prepared_stage_metadata=prepared_stage_metadata,
            )
        else:
          if use_lowmem_active_transition_mode:
            first_transition, first_transition_transpose, transitions, transition_transposes, states = (
              lqr_active_controllable_forward_operators_and_states_lowmem(
                inputs, params, self._layer_apply, self._layer_names, other_model_variables,
                checkpoint_policy=resolved_checkpoint_policy,
              )
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
          if use_grouped_execution_stage_operator_path:
            if use_lowmem_active_k_mode:
              segment_k_operators = lqr_active_segment_backward_hamiltonian_operators_lowmem(
                params, states, final_p, segment_operators, self._execution_stage_apply,
                self._lqr_segment_specs, self._damping, other_model_variables,
                prepared_stage_metadata=prepared_stage_metadata,
                checkpoint_policy=resolved_checkpoint_policy,
                use_fast_paths=self._uses_active_fast_paths(),
                second_order_mode=self._llqr_second_order_mode,
                second_order_chunk_size=second_order_chunk_size,
                batch_axis=second_order_batch_axis,
              )
            else:
              segment_k_operators = lqr_active_segment_backward_hamiltonian_operators(
                params, states, final_p, segment_operators, self._execution_stage_apply,
                self._lqr_segment_specs, self._damping, other_model_variables,
                prepared_stage_metadata=prepared_stage_metadata,
                use_fast_paths=self._uses_active_fast_paths(),
                second_order_mode=self._llqr_second_order_mode,
                second_order_chunk_size=second_order_chunk_size,
                batch_axis=second_order_batch_axis,
              )
            gradients = _recover_loss_gradients_from_lqr_segment_transposes(
              self._layer_names, self._lqr_segment_specs, segment_operators, final_lin_cost,
              self._controlled_stage_unravel_fns, freeze_result=isinstance(params, FrozenDict)
            )
          elif use_lowmem_active_k_mode:
            stage_k_operators = lqr_active_execution_backward_hamiltonian_operators_lowmem(
              params, states, final_p, execution_stage_operators, self._execution_stage_apply,
              self._execution_stage_specs, self._damping, other_model_variables, layer_modules=self._layer_modules,
              prepared_stage_metadata=prepared_stage_metadata,
              checkpoint_policy=resolved_checkpoint_policy,
              use_fast_paths=self._uses_active_fast_paths(),
            )
          else:
            stage_k_operators = lqr_active_execution_backward_hamiltonian_operators(
              params, states, final_p, execution_stage_operators, self._execution_stage_apply,
              self._execution_stage_specs, self._damping, other_model_variables, layer_modules=self._layer_modules,
              prepared_stage_metadata=prepared_stage_metadata,
              use_fast_paths=self._uses_active_fast_paths(),
              batch_update_mode=self._resolved_llqr_batch_update_mode(),
              batch_chunk_size=self._resolved_llqr_batch_update_chunk_size(),
              batch_axis=batch_axis,
            )
          if not use_grouped_execution_stage_operator_path:
            gradients = _recover_loss_gradients_from_execution_stage_transposes(
              self._layer_names, self._execution_stage_specs, execution_stage_operators, final_lin_cost,
              self._controlled_stage_unravel_fns, freeze_result=isinstance(params, FrozenDict)
            )
        else:
          if use_lowmem_active_k_mode:
            first_k_backward, k_backward = lqr_active_controllable_backward_hamiltonian_operators_lowmem(
              params, states, final_p, transition_transposes, self._layer_apply, self._layer_names,
              self._damping, other_model_variables, layer_modules=self._layer_modules,
              checkpoint_policy=resolved_checkpoint_policy,
              use_fast_paths=self._uses_active_fast_paths())
          else:
            first_k_backward, k_backward = lqr_active_controllable_backward_hamiltonian_operators(
              params, states, final_p, transition_transposes, self._layer_apply, self._layer_names,
              self._damping, other_model_variables, layer_modules=self._layer_modules,
              use_fast_paths=self._uses_active_fast_paths(),
              batch_update_mode=self._resolved_llqr_batch_update_mode(),
              batch_chunk_size=self._resolved_llqr_batch_update_chunk_size(),
              batch_axis=batch_axis)
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
          if use_grouped_execution_stage_operator_path:
            return prepared_gradients, (segment_operators, segment_k_operators, final_q, final_lin_cost)
          return prepared_gradients, (execution_stage_operators, stage_k_operators, final_q, final_lin_cost)

        return prepared_gradients, (
          first_transition, first_transition_transpose, transitions, transition_transposes,
          first_k_backward, k_backward, final_q, final_lin_cost)

      # def lqr_cost(_preconditioner, input_size, gradients, kernel_shapes, operators):
      def lqr_cost(_preconditioner, prepared_gradients, operators):
        if use_grouped_execution_stage_operator_path:
          segment_operators, segment_k_operators, final_q, final_lin_cost = operators
          flat_controls = _materialize_active_flat_controls(
            self._block_structure, _preconditioner, self._layer_names, prepared_gradients
          )
          segment_inputs, final_state = _rollout_active_lqr_segments(
            self._lqr_segment_specs, segment_operators, flat_controls
          )
          return _active_lqr_segment_control_cost_from_rollout(
            self._lqr_segment_specs, segment_operators, segment_k_operators,
            flat_controls, segment_inputs, final_state, final_q, final_lin_cost
          )

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
        if use_grouped_execution_stage_operator_path:
          grads = _recover_preconditioner_gradients_from_active_lqr_segment_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, self._lqr_segment_specs,
            prepared_gradients, operators
          )
        elif self._use_execution_stage_active_path:
          grads = _recover_preconditioner_gradients_from_active_execution_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, self._execution_stage_specs,
            prepared_gradients, operators
          )
        else:
          grads = _recover_preconditioner_gradients_from_active_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, prepared_gradients, operators)
        return _sanitize_preconditioner_grads(grads)

      def lqr_value_and_grad_fn(_preconditioner, prepared_gradients, operators):
        if use_grouped_execution_stage_operator_path:
          value, grads = _active_lqr_segment_preconditioner_value_and_grad_from_control_adjoint(
            self._block_structure, _preconditioner, self._layer_names, self._lqr_segment_specs,
            prepared_gradients, operators
          )
        elif self._use_execution_stage_active_path:
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

  def _get_grouped_chunked_update_helpers(self, *, precond_on_update=False):
    def _sanitize_preconditioner_grads(grads):
      return jax.tree_map(Partial(jnp.nan_to_num, nan=0.0, posinf=1.0, neginf=-1.0), grads)

    def build_grouped_segment_problem(params, other_model_variables, datapoint, trainstate_opt_state,
                                      loss_scale, batch_axis, *,
                                      include_segment_k):
      inputs, targets = datapoint
      second_order_batch_axis = batch_axis if self._uses_sample_separable_second_order() else None
      second_order_chunk_size = None
      if self._uses_sample_separable_second_order():
        second_order_chunk_size = self._resolved_second_order_chunk_size(
          fallback_batch_size=_batch_size_from_datapoint(datapoint, second_order_batch_axis)
        )
      prepared_stage_metadata = prepare_active_execution_stage_metadata(
        params,
        self._execution_stage_specs,
        other_model_variables,
        param_unravel_fns=self._controlled_stage_unravel_fns,
        flat_param_sizes=self._controlled_stage_flat_param_sizes,
      ) if self._should_use_shared_active_metadata_runtime() else None
      segment_operators, states = lqr_active_segment_forward_operators_and_states(
        inputs,
        params,
        self._execution_stage_apply,
        self._lqr_segment_specs,
        other_model_variables,
        prepared_stage_metadata=prepared_stage_metadata,
      )
      if self._divergence_args_index is not None:
        div_arg = states[self._divergence_args_index]
      else:
        div_arg = None
      scaled_loss_fn = lambda logits, labels: loss_scale * self._loss_fn(logits, labels)
      final_q, final_p, final_lin_cost = lqr_active_final_costs_and_adjoints(
        scaled_loss_fn,
        states[-1],
        targets,
        div_f=self._divergence_function,
        div_arg=div_arg,
      )

      segment_k_operators = None
      if include_segment_k:
        segment_k_operators = lqr_active_segment_backward_hamiltonian_operators(
          params,
          states,
          final_p,
          segment_operators,
          self._execution_stage_apply,
          self._lqr_segment_specs,
          self._damping,
          other_model_variables,
          prepared_stage_metadata=prepared_stage_metadata,
          use_fast_paths=self._uses_active_fast_paths(),
          second_order_mode=self._llqr_second_order_mode,
          second_order_chunk_size=second_order_chunk_size,
          batch_axis=second_order_batch_axis,
        )

      gradients = _recover_loss_gradients_from_lqr_segment_transposes(
        self._layer_names,
        self._lqr_segment_specs,
        segment_operators,
        final_lin_cost,
        self._controlled_stage_unravel_fns,
        freeze_result=isinstance(params, FrozenDict),
      )
      if precond_on_update:
        gradients, _ = self._trainstate_solver.update(gradients, trainstate_opt_state, params)
      gradients = self._normalize_grad_for_lqr_fn(gradients)
      gradients = jax.tree_map(lambda v: -1.0 * v, gradients)

      operators = None
      if include_segment_k:
        operators = (segment_operators, segment_k_operators, final_q, final_lin_cost)
      return gradients, operators

    @Partial(jax.jit, static_argnums=(5,))
    def get_chunk_loss_gradients(params, other_model_variables, datapoint, trainstate_opt_state,
                                 loss_scale, batch_axis):
      gradients, _ = build_grouped_segment_problem(
        params,
        other_model_variables,
        datapoint,
        trainstate_opt_state,
        loss_scale,
        batch_axis,
        include_segment_k=False,
      )
      return gradients

    @Partial(jax.jit, static_argnums=(7,))
    def get_chunk_grad_only(preconditioner, params, other_model_variables, datapoint,
                            trainstate_opt_state, prepared_gradients, loss_scale, batch_axis):
      _, operators = build_grouped_segment_problem(
        params,
        other_model_variables,
        datapoint,
        trainstate_opt_state,
        loss_scale,
        batch_axis,
        include_segment_k=True,
      )
      grads = _recover_preconditioner_gradients_from_active_lqr_segment_control_adjoint(
        self._block_structure,
        preconditioner,
        self._layer_names,
        self._lqr_segment_specs,
        prepared_gradients,
        operators,
      )
      return _sanitize_preconditioner_grads(grads)

    @Partial(jax.jit, static_argnums=(7,))
    def get_chunk_value_and_grad(preconditioner, params, other_model_variables, datapoint,
                                 trainstate_opt_state, prepared_gradients, loss_scale, batch_axis):
      _, operators = build_grouped_segment_problem(
        params,
        other_model_variables,
        datapoint,
        trainstate_opt_state,
        loss_scale,
        batch_axis,
        include_segment_k=True,
      )
      value, grads = _active_lqr_segment_preconditioner_value_and_grad_from_control_adjoint(
        self._block_structure,
        preconditioner,
        self._layer_names,
        self._lqr_segment_specs,
        prepared_gradients,
        operators,
      )
      return value, _sanitize_preconditioner_grads(grads)

    return get_chunk_loss_gradients, get_chunk_grad_only, get_chunk_value_and_grad

  def _accumulate_grouped_chunked_prepared_gradients(self, params, other_model_variables,
                                                     chunk_datapoints, chunk_weights, batch_axis,
                                                     trainstate_opt_state):
    total_weight = float(sum(chunk_weights))
    if total_weight <= 0:
      raise ValueError("Expected a positive effective loss count for chunked grouped LLQR updates")

    accumulated_gradients = None
    for chunk_datapoint, chunk_weight in zip(chunk_datapoints, chunk_weights):
      gradients = self._grouped_chunked_update_loss_gradients_fn(
        params,
        other_model_variables,
        chunk_datapoint,
        trainstate_opt_state,
        float(chunk_weight) / total_weight,
        batch_axis,
      )
      if accumulated_gradients is None:
        accumulated_gradients = gradients
      else:
        accumulated_gradients = _tree_add(accumulated_gradients, gradients)

    return self._block_structure.prepare_train_vectors(accumulated_gradients)

  def _grouped_chunked_update_grad_only(self, preconditioner, params, other_model_variables,
                                              chunk_datapoints, chunk_weights, batch_axis, trainstate_opt_state,
                                              prepared_gradients):
    total_weight = float(sum(chunk_weights))
    accumulated_grads = None
    for chunk_datapoint, chunk_weight in zip(chunk_datapoints, chunk_weights):
      grads = self._grouped_chunked_update_grad_only_fn(
        preconditioner,
        params,
        other_model_variables,
        chunk_datapoint,
        trainstate_opt_state,
        prepared_gradients,
        float(chunk_weight) / total_weight,
        batch_axis,
      )
      if accumulated_grads is None:
        accumulated_grads = grads
      else:
        accumulated_grads = _tree_add(accumulated_grads, grads)
    return accumulated_grads

  def _grouped_chunked_update_value_and_grad(self, preconditioner, params, other_model_variables,
                                                   chunk_datapoints, chunk_weights, batch_axis, trainstate_opt_state,
                                                   prepared_gradients):
    total_weight = float(sum(chunk_weights))
    accumulated_value = 0.0
    accumulated_grads = None
    for chunk_datapoint, chunk_weight in zip(chunk_datapoints, chunk_weights):
      value, grads = self._grouped_chunked_update_value_and_grad_fn(
        preconditioner,
        params,
        other_model_variables,
        chunk_datapoint,
        trainstate_opt_state,
        prepared_gradients,
        float(chunk_weight) / total_weight,
        batch_axis,
      )
      accumulated_value += value
      if accumulated_grads is None:
        accumulated_grads = grads
      else:
        accumulated_grads = _tree_add(accumulated_grads, grads)
    return accumulated_value, accumulated_grads

  def _run_grouped_chunked_update(self, preconditioner, params, precond_lr, other_model_variables,
                                  chunk_datapoints, chunk_weights, batch_axis, trainstate_opt_state, *,
                                  snapshot_preconditioner):
    if snapshot_preconditioner:
      current_preconditioner = _deep_copy_pytree(preconditioner)
    else:
      current_preconditioner = preconditioner

    prepared_gradients = self._accumulate_grouped_chunked_prepared_gradients(
      params,
      other_model_variables,
      chunk_datapoints,
      chunk_weights,
      batch_axis,
      trainstate_opt_state,
    )
    opt_state = self._optax_solver.init(current_preconditioner)

    for _ in range(self._preconditioner_update_steps):
      if self._optax_solver_requires_value_and_grad:
        def chunked_value_and_grad_fn(local_preconditioner):
          return self._grouped_chunked_update_value_and_grad(
            local_preconditioner,
            params,
            other_model_variables,
            chunk_datapoints,
            chunk_weights,
            batch_axis,
            trainstate_opt_state,
            prepared_gradients,
          )

        _, preconditioner_grad = chunked_value_and_grad_fn(current_preconditioner)
        updates, opt_state = self._optax_solver.update(
          preconditioner_grad,
          opt_state,
          current_preconditioner,
          value_and_grad_fn=chunked_value_and_grad_fn,
        )
      else:
        preconditioner_grad = self._grouped_chunked_update_grad_only(
          current_preconditioner,
          params,
          other_model_variables,
          chunk_datapoints,
          chunk_weights,
          batch_axis,
          trainstate_opt_state,
          prepared_gradients,
        )
        updates, opt_state = self._optax_solver.update(
          preconditioner_grad,
          opt_state,
          current_preconditioner,
        )
      updates = jax.tree_map(lambda g: g * precond_lr, updates)
      current_preconditioner = optax.apply_updates(current_preconditioner, updates)

    return jax.tree_map(jnp.nan_to_num, current_preconditioner)

  def update_preconditioner(self, params, dataloader, precond_lr, opt_state, precond_batch_size, ema_decay=0, other_model_variables=FrozenDict({})):
    """params is the current weights of the NN"""
    if self._multibatch:
      self._record_llqr_batch_update_route(
        operation="update",
        route="multibatch",
        precond_batch_size=precond_batch_size,
      )
      self._block_structure.update_blocks(
        self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables, dataloader,
                                       opt_state), ema_decay)
    else:
      if self._warm_start_precond and not hasattr(self._block_structure, "_memory"):
        _blocks = self._block_structure.blocks
      else:
        _blocks = self._block_structure.reinit_blocks()
      should_snapshot = self._should_snapshot_preconditioner_for_update(ema_decay)
      if self._uses_chunked_lqr_segment_update_path():
        layout, chunk_datapoints, chunk_weights = _take_preconditioner_chunk_datapoints(
          dataloader,
          precond_batch_size,
          self._resolved_llqr_batch_update_chunk_size(),
        )
        if _supports_chunked_execution_stage_update_layout(layout):
          normalized_layout, normalized_chunk_datapoints = _normalize_execution_stage_update_chunks(
            chunk_datapoints,
            layout,
          )
          self._record_llqr_batch_update_route(
            operation="update",
            route="chunked_lqr_segment",
            precond_batch_size=precond_batch_size,
            layout=layout,
            normalized_layout=normalized_layout,
            chunk_datapoints=normalized_chunk_datapoints,
            chunk_weights=chunk_weights,
          )
          updated_blocks = self._run_grouped_chunked_update(
            _blocks,
            params,
            precond_lr,
            other_model_variables,
            normalized_chunk_datapoints,
            chunk_weights,
            normalized_layout["batch_axis"],
            opt_state,
            snapshot_preconditioner=should_snapshot,
          )
        else:
          acc_batches = _concat_preconditioner_batches(chunk_datapoints, layout, precond_batch_size)
          self._record_llqr_batch_update_route(
            operation="update",
            route="chunked_lqr_segment_layout_fallback",
            precond_batch_size=precond_batch_size,
            layout=layout,
            chunk_datapoints=chunk_datapoints,
            chunk_weights=chunk_weights,
            fallback_reason=f"unsupported_layout:{layout.get('mode')}",
          )
          updated_blocks = self._update_preconditioner_fn(
            _blocks,
            params,
            precond_lr,
            other_model_variables,
            acc_batches,
            opt_state,
            snapshot_preconditioner=should_snapshot,
          )
      else:
        layout, acc_batches = _take_full_preconditioner_datapoint(dataloader, precond_batch_size)
        normalized_layout = layout
        if self._use_execution_stage_active_path:
          normalized_layout, acc_batches = _normalize_execution_stage_update_datapoint(acc_batches, layout)
        self._record_llqr_batch_update_route(
          operation="update",
          route="full_batch",
          precond_batch_size=precond_batch_size,
          layout=layout,
          normalized_layout=normalized_layout,
        )
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
        self._record_llqr_batch_update_route(
          operation="compile",
          route="multibatch",
          precond_batch_size=precond_batch_size,
        )
        blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables, dataloader,
                                       opt_state)
    else:
      if self._uses_chunked_lqr_segment_update_path():
        layout, chunk_datapoints, chunk_weights = _take_preconditioner_chunk_datapoints(
          dataloader,
          precond_batch_size,
          self._resolved_llqr_batch_update_chunk_size(),
        )
        if _supports_chunked_execution_stage_update_layout(layout):
          normalized_layout, normalized_chunk_datapoints = _normalize_execution_stage_update_chunks(
            chunk_datapoints,
            layout,
          )
          self._record_llqr_batch_update_route(
            operation="compile",
            route="chunked_lqr_segment",
            precond_batch_size=precond_batch_size,
            layout=layout,
            normalized_layout=normalized_layout,
            chunk_datapoints=normalized_chunk_datapoints,
            chunk_weights=chunk_weights,
          )
          blocks = self._run_grouped_chunked_update(
            self._block_structure.blocks,
            params,
            precond_lr,
            other_model_variables,
            normalized_chunk_datapoints,
            chunk_weights,
            normalized_layout["batch_axis"],
            opt_state,
            snapshot_preconditioner=True,
          )
        else:
          acc_batches = _concat_preconditioner_batches(chunk_datapoints, layout, precond_batch_size)
          self._record_llqr_batch_update_route(
            operation="compile",
            route="chunked_lqr_segment_layout_fallback",
            precond_batch_size=precond_batch_size,
            layout=layout,
            chunk_datapoints=chunk_datapoints,
            chunk_weights=chunk_weights,
            fallback_reason=f"unsupported_layout:{layout.get('mode')}",
          )
          blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables,
                                     acc_batches, opt_state)
      else:
        layout, acc_batches = _take_full_preconditioner_datapoint(dataloader, precond_batch_size)
        normalized_layout = layout
        if self._use_execution_stage_active_path:
          normalized_layout, acc_batches = _normalize_execution_stage_update_datapoint(acc_batches, layout)
        self._record_llqr_batch_update_route(
          operation="compile",
          route="full_batch",
          precond_batch_size=precond_batch_size,
          layout=layout,
          normalized_layout=normalized_layout,
        )
        blocks = self._update_preconditioner_fn(self._block_structure.blocks, params, precond_lr, other_model_variables,
                                     acc_batches, opt_state)

  def expose_blocks(self):
    return self._block_structure.blocks

  def load_blocks(self, saved_blocks):
    self._block_structure.update_blocks(saved_blocks)
