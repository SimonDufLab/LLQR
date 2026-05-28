"""Grouped LLQR segment builders for split execution-stage models."""
import numbers
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from lqr_optimizer._src.utils.build_lqr import (
  _cast_state_like,
  _maybe_checkpoint,
  _state_primal_for_linearization,
  prepare_active_execution_stage_metadata,
  tree_vdot,
  zero_tangent_tree_like,
)
from lqr_optimizer._src.utils.sample_separable_second_order import (
  build_sample_separable_second_order_actions,
  build_sample_separable_state_only_action,
)


VALID_LQR_SEGMENT_SECOND_ORDER_MODES = ("batched_exact", "sample_separable_exact")


class ActiveLqrSegmentOperator(NamedTuple):
  kind: str
  segment_name: str
  controlled_param_names: tuple
  forward: callable
  transpose: callable


def _execution_stage_specs_from_segments(lqr_segment_specs):
  return tuple(
    stage_spec
    for segment_spec in lqr_segment_specs
    for stage_spec in segment_spec.execution_stage_descriptors
  )


def _segment_stage_metadata(prepared_stage_metadata, segment_spec):
  return tuple(prepared_stage_metadata[segment_spec.start_index:segment_spec.stop_index])


def _segment_flat_params(segment_stage_metadata):
  return {
    stage_metadata.stage_spec.param_name: stage_metadata.flat_params
    for stage_metadata in segment_stage_metadata
    if stage_metadata.stage_spec.kind == "controlled"
  }


def _add_damping_to_segment_controls(control_action, control_tangent, damping):
  return {
    name: jnp.ravel(jnp.atleast_1d(control_action[name])) + damping * jnp.ravel(control_tangent[name])
    for name in control_tangent
  }


def _add_segment_control_actions(lhs, rhs):
  return {
    name: jnp.ravel(jnp.atleast_1d(lhs[name])) + jnp.ravel(jnp.atleast_1d(rhs[name]))
    for name in lhs
  }


def _validate_lqr_segment_second_order_options(second_order_mode, second_order_chunk_size, batch_axis):
  if second_order_mode not in VALID_LQR_SEGMENT_SECOND_ORDER_MODES:
    raise ValueError(
      f"Unknown second_order_mode '{second_order_mode}'. "
      f"Expected one of {VALID_LQR_SEGMENT_SECOND_ORDER_MODES}."
    )
  if second_order_mode == "batched_exact":
    if second_order_chunk_size is not None:
      raise ValueError("second_order_chunk_size must be None when second_order_mode='batched_exact'.")
    return second_order_mode, None, None

  if (
      not isinstance(second_order_chunk_size, numbers.Integral)
      or isinstance(second_order_chunk_size, bool)
      or int(second_order_chunk_size) <= 0
  ):
    raise ValueError(
      "second_order_chunk_size must be a positive integer when "
      "second_order_mode='sample_separable_exact'."
    )
  if not isinstance(batch_axis, numbers.Integral) or isinstance(batch_axis, bool) or int(batch_axis) < 0:
    raise ValueError("batch_axis must be a non-negative integer when second_order_mode='sample_separable_exact'.")
  return second_order_mode, int(second_order_chunk_size), int(batch_axis)


def describe_lqr_segment_sample_separable_support(lqr_segment_specs):
  unsupported_segments = []
  supported_count = 0
  for segment_spec in lqr_segment_specs:
    policy = getattr(segment_spec, "sample_separable_second_order", None)
    if policy is True:
      supported_count += 1
      continue
    reason = "metadata_missing" if policy is None else "metadata_false"
    unsupported_segments.append({"name": segment_spec.name, "reason": reason})
  return {
    "sample_separable_supported_segment_count": supported_count,
    "sample_separable_unsupported_segments": unsupported_segments,
  }


def _validate_sample_separable_segment_support(lqr_segment_specs):
  support = describe_lqr_segment_sample_separable_support(lqr_segment_specs)
  if support["sample_separable_unsupported_segments"]:
    unsupported = ", ".join(
      f"{item['name']}:{item['reason']}"
      for item in support["sample_separable_unsupported_segments"]
    )
    raise ValueError(
      "sample_separable_exact requires every LLQR segment to declare "
      f"sample-separable second-order support; unsupported segments: {unsupported}"
    )
  return support


def describe_lqr_segment_second_order_route(lqr_segment_specs, *, second_order_mode="batched_exact",
                                            second_order_chunk_size=None, batch_axis=None):
  second_order_mode, second_order_chunk_size, batch_axis = _validate_lqr_segment_second_order_options(
    second_order_mode, second_order_chunk_size, batch_axis
  )
  support = describe_lqr_segment_sample_separable_support(lqr_segment_specs)
  if second_order_mode == "sample_separable_exact":
    _validate_sample_separable_segment_support(lqr_segment_specs)
  return {
    "second_order_mode": second_order_mode,
    "second_order_chunk_size": second_order_chunk_size,
    "second_order_batch_axis": batch_axis,
    "uses_sample_separable_second_order": second_order_mode == "sample_separable_exact",
    "lqr_segment_count": len(lqr_segment_specs),
    "controlled_param_count": sum(len(segment.controlled_param_names) for segment in lqr_segment_specs),
    **support,
  }


def _apply_segment(stages_apply, segment_stage_metadata, segment_input_state, segment_params, segment_start_index):
  x = segment_input_state
  for local_index, stage_metadata in enumerate(segment_stage_metadata):
    execution_index = segment_start_index + local_index
    stage_spec = stage_metadata.stage_spec
    if stage_spec.kind == "controlled":
      flat_params = segment_params[stage_spec.param_name]
      variables = {"params": stage_metadata.unravel_params_fn(flat_params)} | stage_metadata.other_vars
      x = stages_apply(variables, x, execution_index)
    else:
      x = stages_apply(stage_metadata.other_vars, x, execution_index)
  return x


def _build_segment_control_only_transition_operator(segment_params, simpler_apply):
  _, control_linear = jax.linearize(simpler_apply, segment_params)
  control_transpose = jax.linear_transpose(control_linear, segment_params)

  def transition(control_tangent):
    return control_linear(control_tangent)

  def transition_transpose(cotangent):
    param_cotangent = control_transpose(cotangent)
    if isinstance(param_cotangent, tuple):
      param_cotangent = param_cotangent[0]
    return {
      name: jnp.ravel(jnp.atleast_1d(param_cotangent[name]))
      for name in segment_params
    }

  return transition, transition_transpose


def _build_segment_joint_transition_operator(segment_params, segment_state, simpler_apply):
  segment_state_primal = _state_primal_for_linearization(segment_state)

  def apply_from_linear_state(parameters, state):
    return simpler_apply(parameters, _cast_state_like(segment_state, state))

  _, joint_linear = jax.linearize(apply_from_linear_state, segment_params, segment_state_primal)
  joint_transpose = jax.linear_transpose(joint_linear, segment_params, segment_state_primal)

  def transition(control_tangent, state_tangent):
    return joint_linear(control_tangent, state_tangent)

  def transition_transpose(cotangent):
    param_cotangent, state_cotangent = joint_transpose(cotangent)
    return {
      name: jnp.ravel(jnp.atleast_1d(param_cotangent[name]))
      for name in segment_params
    }, state_cotangent

  return transition, transition_transpose


def _build_segment_state_only_transition_operator(segment_state, simpler_apply):
  segment_state_primal = _state_primal_for_linearization(segment_state)

  def apply_from_linear_state(state):
    return simpler_apply(_cast_state_like(segment_state, state))

  _, state_linear = jax.linearize(apply_from_linear_state, segment_state_primal)
  state_transpose = jax.linear_transpose(state_linear, segment_state_primal)

  def transition(state_tangent):
    return state_linear(state_tangent)

  def transition_transpose(cotangent):
    state_cotangent = state_transpose(cotangent)
    if isinstance(state_cotangent, tuple):
      state_cotangent = state_cotangent[0]
    return state_cotangent

  return transition, transition_transpose


def _build_on_demand_segment_control_only_transition_operator(segment_params, simpler_apply,
                                                             checkpoint_policy="none"):
  apply_from_control = _maybe_checkpoint(simpler_apply, checkpoint_policy)

  def transition(control_tangent, apply_from_control=apply_from_control, segment_params=segment_params):
    _, output_tangent = jax.jvp(apply_from_control, (segment_params,), (control_tangent,))
    return output_tangent

  def transition_transpose(cotangent, apply_from_control=apply_from_control, segment_params=segment_params):
    _, pullback = jax.vjp(apply_from_control, segment_params)
    param_cotangent = pullback(cotangent)
    if isinstance(param_cotangent, tuple):
      param_cotangent = param_cotangent[0]
    return {
      name: jnp.ravel(jnp.atleast_1d(param_cotangent[name]))
      for name in segment_params
    }

  return transition, transition_transpose


def _build_on_demand_segment_joint_transition_operator(segment_params, segment_state, simpler_apply,
                                                       checkpoint_policy="none"):
  segment_state_primal = _state_primal_for_linearization(segment_state)
  apply_from_linear_state = _maybe_checkpoint(simpler_apply, checkpoint_policy)

  def transition(control_tangent, state_tangent, apply_from_linear_state=apply_from_linear_state,
                 segment_params=segment_params, segment_state_primal=segment_state_primal):
    _, output_tangent = jax.jvp(
      apply_from_linear_state,
      (segment_params, segment_state_primal),
      (control_tangent, state_tangent),
    )
    return output_tangent

  def transition_transpose(cotangent, apply_from_linear_state=apply_from_linear_state,
                           segment_params=segment_params, segment_state_primal=segment_state_primal):
    _, pullback = jax.vjp(apply_from_linear_state, segment_params, segment_state_primal)
    param_cotangent, state_cotangent = pullback(cotangent)
    return {
      name: jnp.ravel(jnp.atleast_1d(param_cotangent[name]))
      for name in segment_params
    }, state_cotangent

  return transition, transition_transpose


def _build_on_demand_segment_state_only_transition_operator(segment_state, simpler_apply,
                                                           checkpoint_policy="none"):
  segment_state_primal = _state_primal_for_linearization(segment_state)
  apply_from_linear_state = _maybe_checkpoint(simpler_apply, checkpoint_policy)

  def transition(state_tangent, apply_from_linear_state=apply_from_linear_state,
                 segment_state_primal=segment_state_primal):
    _, output_tangent = jax.jvp(apply_from_linear_state, (segment_state_primal,), (state_tangent,))
    return output_tangent

  def transition_transpose(cotangent, apply_from_linear_state=apply_from_linear_state,
                           segment_state_primal=segment_state_primal):
    _, pullback = jax.vjp(apply_from_linear_state, segment_state_primal)
    state_cotangent = pullback(cotangent)
    if isinstance(state_cotangent, tuple):
      state_cotangent = state_cotangent[0]
    return state_cotangent

  return transition, transition_transpose


def _lqr_active_segment_forward_operators_and_states(batch, params, stages_apply, lqr_segment_specs,
                                                     other_model_variables=FrozenDict({}),
                                                     prepared_stage_metadata=None,
                                                     checkpoint_policy="none",
                                                     lowmem=False):
  if not lqr_segment_specs:
    raise ValueError("Grouped LLQR segment builders expect at least one segment.")
  if prepared_stage_metadata is None:
    prepared_stage_metadata = prepare_active_execution_stage_metadata(
      params,
      _execution_stage_specs_from_segments(lqr_segment_specs),
      other_model_variables,
    )

  states = [batch]
  segment_operators = []
  seen_control = False

  for segment_spec in lqr_segment_specs:
    segment_metadata = _segment_stage_metadata(prepared_stage_metadata, segment_spec)
    segment_params = _segment_flat_params(segment_metadata)
    segment_state = states[-1]

    if segment_params:
      if not seen_control:
        fixed_input_state = segment_state

        def simpler_apply(parameters, segment_metadata=segment_metadata,
                          fixed_input_state=fixed_input_state,
                          segment_start_index=segment_spec.start_index):
          return _apply_segment(
            stages_apply, segment_metadata, fixed_input_state, parameters, segment_start_index)

        states.append(simpler_apply(segment_params))
        if lowmem:
          forward_op, transpose_op = _build_on_demand_segment_control_only_transition_operator(
            segment_params, simpler_apply, checkpoint_policy=checkpoint_policy)
        else:
          forward_op, transpose_op = _build_segment_control_only_transition_operator(
            segment_params, simpler_apply)
        segment_operators.append(
          ActiveLqrSegmentOperator(
            "control_only", segment_spec.name, segment_spec.controlled_param_names, forward_op, transpose_op
          )
        )
        seen_control = True
      else:
        def simpler_apply(parameters, x, segment_metadata=segment_metadata,
                          segment_state=segment_state, segment_start_index=segment_spec.start_index):
          return _apply_segment(
            stages_apply,
            segment_metadata,
            _cast_state_like(segment_state, x),
            parameters,
            segment_start_index,
          )

        states.append(simpler_apply(segment_params, segment_state))
        if lowmem:
          forward_op, transpose_op = _build_on_demand_segment_joint_transition_operator(
            segment_params, segment_state, simpler_apply, checkpoint_policy=checkpoint_policy)
        else:
          forward_op, transpose_op = _build_segment_joint_transition_operator(
            segment_params, segment_state, simpler_apply)
        segment_operators.append(
          ActiveLqrSegmentOperator(
            "controlled", segment_spec.name, segment_spec.controlled_param_names, forward_op, transpose_op
          )
        )
    else:
      if not seen_control:
        raise ValueError("Passive LLQR segments before the first controlled segment are not supported.")

      def simpler_apply(x, segment_metadata=segment_metadata,
                        segment_state=segment_state, segment_start_index=segment_spec.start_index):
        return _apply_segment(
          stages_apply, segment_metadata, _cast_state_like(segment_state, x), {}, segment_start_index)

      states.append(simpler_apply(segment_state))
      if lowmem:
        forward_op, transpose_op = _build_on_demand_segment_state_only_transition_operator(
          segment_state, simpler_apply, checkpoint_policy=checkpoint_policy)
      else:
        forward_op, transpose_op = _build_segment_state_only_transition_operator(
          segment_state, simpler_apply)
      segment_operators.append(
        ActiveLqrSegmentOperator("passive", segment_spec.name, (), forward_op, transpose_op)
      )

  return segment_operators, states


def lqr_active_segment_forward_operators_and_states(batch, params, stages_apply, lqr_segment_specs,
                                                    other_model_variables=FrozenDict({}),
                                                    prepared_stage_metadata=None):
  return _lqr_active_segment_forward_operators_and_states(
    batch,
    params,
    stages_apply,
    lqr_segment_specs,
    other_model_variables,
    prepared_stage_metadata=prepared_stage_metadata,
    lowmem=False,
  )


def lqr_active_segment_forward_operators_and_states_lowmem(batch, params, stages_apply, lqr_segment_specs,
                                                           other_model_variables=FrozenDict({}),
                                                           prepared_stage_metadata=None,
                                                           checkpoint_policy="none"):
  return _lqr_active_segment_forward_operators_and_states(
    batch,
    params,
    stages_apply,
    lqr_segment_specs,
    other_model_variables,
    prepared_stage_metadata=prepared_stage_metadata,
    checkpoint_policy=checkpoint_policy,
    lowmem=True,
  )


def _lqr_active_segment_backward_hamiltonian_operators(params, states, final_adjoint, segment_operators,
                                                       stages_apply, lqr_segment_specs, damping,
                                                       other_model_variables=FrozenDict({}),
                                                       prepared_stage_metadata=None,
                                                       checkpoint_policy="none",
                                                       lowmem=False,
                                                       second_order_mode="batched_exact",
                                                       second_order_chunk_size=None,
                                                       batch_axis=None):
  if not lqr_segment_specs:
    raise ValueError("Grouped LLQR segment K builders expect at least one segment.")
  second_order_mode, second_order_chunk_size, batch_axis = _validate_lqr_segment_second_order_options(
    second_order_mode, second_order_chunk_size, batch_axis
  )
  if second_order_mode == "sample_separable_exact":
    _validate_sample_separable_segment_support(lqr_segment_specs)
  if prepared_stage_metadata is None:
    prepared_stage_metadata = prepare_active_execution_stage_metadata(
      params,
      _execution_stage_specs_from_segments(lqr_segment_specs),
      other_model_variables,
    )

  p_i = final_adjoint
  k_rev = []
  for reverse_index, segment_spec in enumerate(reversed(lqr_segment_specs)):
    segment_index = len(lqr_segment_specs) - reverse_index - 1
    segment_operator = segment_operators[segment_index]
    segment_metadata = _segment_stage_metadata(prepared_stage_metadata, segment_spec)
    segment_params = _segment_flat_params(segment_metadata)
    segment_state = states[segment_index]

    if segment_operator.kind == "passive":
      def simpler_apply(x, segment_metadata=segment_metadata,
                        segment_state=segment_state, segment_start_index=segment_spec.start_index):
        return _apply_segment(
          stages_apply, segment_metadata, _cast_state_like(segment_state, x), {}, segment_start_index)

      if all(stage.passive_state_hessian == "zero" for stage in segment_spec.execution_stage_descriptors):
        def k_i(state_tangent):
          return zero_tangent_tree_like(state_tangent)
      elif second_order_mode == "sample_separable_exact":
        sample_state_action = build_sample_separable_state_only_action(
          simpler_apply,
          segment_state,
          p_i,
          batch_axis=batch_axis,
          second_order_chunk_size=second_order_chunk_size,
        )

        def k_i(state_tangent, sample_state_action=sample_state_action):
          return sample_state_action(state_tangent)
      else:
        segment_state_primal = _state_primal_for_linearization(segment_state)
        if lowmem:
          checkpointed_apply = _maybe_checkpoint(simpler_apply, checkpoint_policy)

          def hamiltonian_x(x, checkpointed_apply=checkpointed_apply, p_i=p_i):
            return tree_vdot(checkpointed_apply(x), p_i)

          grad_hamiltonian = jax.grad(hamiltonian_x)

          def k_i(state_tangent, grad_hamiltonian=grad_hamiltonian,
                  segment_state_primal=segment_state_primal):
            _, hessian_action = jax.jvp(grad_hamiltonian, (segment_state_primal,), (state_tangent,))
            return hessian_action
        else:
          def hamiltonian_x(x, p_i=p_i):
            return tree_vdot(simpler_apply(x), p_i)

          _, state_hessian = jax.linearize(jax.grad(hamiltonian_x), segment_state_primal)

          def k_i(state_tangent, state_hessian=state_hessian):
            return state_hessian(state_tangent)

      k_rev.append(k_i)
      p_i = segment_operator.transpose(p_i)
      continue

    if segment_operator.kind == "control_only":
      fixed_input_state = segment_state

      def simpler_apply(parameters, segment_metadata=segment_metadata,
                        fixed_input_state=fixed_input_state,
                        segment_start_index=segment_spec.start_index):
        return _apply_segment(
          stages_apply, segment_metadata, fixed_input_state, parameters, segment_start_index)

      if second_order_mode == "sample_separable_exact":
        def sample_apply(parameters, x, segment_metadata=segment_metadata,
                         fixed_input_state=fixed_input_state,
                         segment_start_index=segment_spec.start_index):
          return _apply_segment(
            stages_apply,
            segment_metadata,
            _cast_state_like(fixed_input_state, x),
            parameters,
            segment_start_index,
          )

        sample_actions = build_sample_separable_second_order_actions(
          sample_apply,
          segment_params,
          fixed_input_state,
          p_i,
          batch_axis=batch_axis,
          second_order_chunk_size=second_order_chunk_size,
          damping=damping,
        )

        def k_i(control_tangent, sample_actions=sample_actions):
          return sample_actions.r(control_tangent)
      elif lowmem:
        checkpointed_apply = _maybe_checkpoint(simpler_apply, checkpoint_policy)

        def hamiltonian(parameters, checkpointed_apply=checkpointed_apply, p_i=p_i):
          return tree_vdot(checkpointed_apply(parameters), p_i)

        grad_hamiltonian = jax.grad(hamiltonian)

        def k_i(control_tangent, grad_hamiltonian=grad_hamiltonian,
                segment_params=segment_params, damping=damping):
          _, hessian_action = jax.jvp(grad_hamiltonian, (segment_params,), (control_tangent,))
          return _add_damping_to_segment_controls(hessian_action, control_tangent, damping)
      else:
        def hamiltonian(parameters, p_i=p_i):
          return tree_vdot(simpler_apply(parameters), p_i)

        _, first_r = jax.linearize(jax.grad(hamiltonian), segment_params)

        def k_i(control_tangent, first_r=first_r, damping=damping):
          return _add_damping_to_segment_controls(first_r(control_tangent), control_tangent, damping)

      k_rev.append(k_i)
      continue

    segment_state_primal = _state_primal_for_linearization(segment_state)

    def simpler_apply(parameters, x, segment_metadata=segment_metadata,
                      segment_state=segment_state, segment_start_index=segment_spec.start_index):
      return _apply_segment(
        stages_apply,
        segment_metadata,
        _cast_state_like(segment_state, x),
        parameters,
        segment_start_index,
      )

    if second_order_mode == "sample_separable_exact":
      sample_actions = build_sample_separable_second_order_actions(
        simpler_apply,
        segment_params,
        segment_state,
        p_i,
        batch_axis=batch_axis,
        second_order_chunk_size=second_order_chunk_size,
        damping=damping,
      )

      def k_i(control_tangent, state_tangent, sample_actions=sample_actions):
        control_action = _add_segment_control_actions(
          sample_actions.r(control_tangent),
          sample_actions.m(state_tangent),
        )
        state_action = jax.tree_util.tree_map(
          jnp.add,
          sample_actions.mt(control_tangent),
          sample_actions.q(state_tangent),
        )
        return control_action, state_action
    elif lowmem:
      checkpointed_apply = _maybe_checkpoint(simpler_apply, checkpoint_policy)

      def hamiltonian_joint(parameters, x, checkpointed_apply=checkpointed_apply, p_i=p_i):
        return tree_vdot(checkpointed_apply(parameters, x), p_i)

      grad_hamiltonian = jax.grad(hamiltonian_joint, argnums=(0, 1))

      def k_i(control_tangent, state_tangent, grad_hamiltonian=grad_hamiltonian,
              segment_params=segment_params, segment_state_primal=segment_state_primal,
              damping=damping):
        _, hessian_action = jax.jvp(
          grad_hamiltonian,
          (segment_params, segment_state_primal),
          (control_tangent, state_tangent),
        )
        control_action, state_action = hessian_action
        return _add_damping_to_segment_controls(control_action, control_tangent, damping), state_action
    else:
      def hamiltonian_joint(parameters, x, p_i=p_i):
        return tree_vdot(simpler_apply(parameters, x), p_i)

      _, joint_hessian = jax.linearize(
        jax.grad(hamiltonian_joint, argnums=(0, 1)), segment_params, segment_state_primal)

      def k_i(control_tangent, state_tangent, joint_hessian=joint_hessian, damping=damping):
        control_action, state_action = joint_hessian(control_tangent, state_tangent)
        return _add_damping_to_segment_controls(control_action, control_tangent, damping), state_action

    k_rev.append(k_i)
    _, p_i = segment_operator.transpose(p_i)

  return list(reversed(k_rev))


def lqr_active_segment_backward_hamiltonian_operators(params, states, final_adjoint, segment_operators,
                                                      stages_apply, lqr_segment_specs, damping,
                                                      other_model_variables=FrozenDict({}),
                                                      prepared_stage_metadata=None,
                                                      use_fast_paths=True,
                                                      second_order_mode="batched_exact",
                                                      second_order_chunk_size=None,
                                                      batch_axis=None):
  del use_fast_paths
  return _lqr_active_segment_backward_hamiltonian_operators(
    params,
    states,
    final_adjoint,
    segment_operators,
    stages_apply,
    lqr_segment_specs,
    damping,
    other_model_variables,
    prepared_stage_metadata=prepared_stage_metadata,
    second_order_mode=second_order_mode,
    second_order_chunk_size=second_order_chunk_size,
    batch_axis=batch_axis,
    lowmem=False,
  )


def lqr_active_segment_backward_hamiltonian_operators_lowmem(params, states, final_adjoint, segment_operators,
                                                             stages_apply, lqr_segment_specs, damping,
                                                             other_model_variables=FrozenDict({}),
                                                             prepared_stage_metadata=None,
                                                             checkpoint_policy="none",
                                                             use_fast_paths=True,
                                                             second_order_mode="batched_exact",
                                                             second_order_chunk_size=None,
                                                             batch_axis=None):
  del use_fast_paths
  return _lqr_active_segment_backward_hamiltonian_operators(
    params,
    states,
    final_adjoint,
    segment_operators,
    stages_apply,
    lqr_segment_specs,
    damping,
    other_model_variables,
    prepared_stage_metadata=prepared_stage_metadata,
    checkpoint_policy=checkpoint_policy,
    second_order_mode=second_order_mode,
    second_order_chunk_size=second_order_chunk_size,
    batch_axis=batch_axis,
    lowmem=True,
  )
