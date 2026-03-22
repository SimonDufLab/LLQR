"""Helper functions for building the LQR problem associated with the desired divergence measure"""
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.core.frozen_dict import FrozenDict

from lqr_optimizer._src.models.mlp import DenseRelu, InitDenseRelu
from lqr_optimizer._src.utils.divergence import ngd_divergence_f
from lqr_optimizer._src.utils.utils import vjp_f, add_f, cross_entropy_loss, StageDescriptor


def diag_r(penalty):
  return lambda v: penalty * v  # Equivalent to having R_i as an identity matrix x constant


def tree_vdot(lhs, rhs):
  lhs_leaves = jax.tree_util.tree_leaves(lhs)
  rhs_leaves = jax.tree_util.tree_leaves(rhs)
  if len(lhs_leaves) != len(rhs_leaves):
    raise ValueError("tree_vdot expects matching pytree structures")
  return sum(jnp.vdot(lhs_leaf, rhs_leaf) for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves))


def zero_tangent_tree_like(state):
  def zeros_like_tangent(leaf):
    dtype = leaf.dtype if jnp.issubdtype(leaf.dtype, jnp.inexact) else jnp.float32
    return jnp.zeros(leaf.shape, dtype=dtype)

  return jax.tree_util.tree_map(zeros_like_tangent, state)


def _state_primal_for_linearization(state):
  def promote_leaf(leaf):
    return leaf if jnp.issubdtype(leaf.dtype, jnp.inexact) else leaf.astype(jnp.float32)

  return jax.tree_util.tree_map(promote_leaf, state)


def _cast_state_like(reference_state, state):
  def cast_leaf(reference_leaf, state_leaf):
    if reference_leaf.dtype == state_leaf.dtype:
      return state_leaf
    return state_leaf.astype(reference_leaf.dtype)

  return jax.tree_util.tree_map(cast_leaf, reference_state, state)


def _build_joint_transition_operator(layer_params, layer_state, simpler_apply):
  """Build a single linearized transition operator and its transpose for one layer."""
  layer_state_primal = _state_primal_for_linearization(layer_state)

  def apply_from_linear_state(parameters, state):
    return simpler_apply(parameters, _cast_state_like(layer_state, state))

  _, joint_linear = jax.linearize(apply_from_linear_state, layer_params, layer_state_primal)
  joint_transpose = jax.linear_transpose(joint_linear, layer_params, layer_state_primal)

  def transition(control_tangent, state_tangent):
    return joint_linear(jnp.ravel(control_tangent), state_tangent)

  def transition_transpose(cotangent):
    param_cotangent, state_cotangent = joint_transpose(cotangent)
    return jnp.ravel(jnp.atleast_1d(param_cotangent)), state_cotangent

  return transition, transition_transpose


def _build_control_only_transition_operator(layer_params, simpler_apply):
  """Build a control-only transition operator with a fixed input state."""
  _, control_linear = jax.linearize(simpler_apply, layer_params)
  control_transpose = jax.linear_transpose(control_linear, layer_params)

  def transition(control_tangent):
    return control_linear(jnp.ravel(control_tangent))

  def transition_transpose(cotangent):
    param_cotangent = control_transpose(cotangent)
    if isinstance(param_cotangent, tuple):
      param_cotangent = param_cotangent[0]
    return jnp.ravel(jnp.atleast_1d(param_cotangent))

  return transition, transition_transpose


def _build_state_only_transition_operator(layer_state, simpler_apply):
  layer_state_primal = _state_primal_for_linearization(layer_state)

  def apply_from_linear_state(state):
    return simpler_apply(_cast_state_like(layer_state, state))

  _, state_linear = jax.linearize(apply_from_linear_state, layer_state_primal)
  state_transpose = jax.linear_transpose(state_linear, layer_state_primal)

  def transition(state_tangent):
    return state_linear(state_tangent)

  def transition_transpose(cotangent):
    state_cotangent = state_transpose(cotangent)
    if isinstance(state_cotangent, tuple):
      state_cotangent = state_cotangent[0]
    return state_cotangent

  return transition, transition_transpose


class ActiveExecutionStageOperator(NamedTuple):
  kind: str
  stage_name: str
  param_name: Optional[str]
  forward: callable
  transpose: callable


def _supports_piecewise_linear_active_fast_path(layer_module):
  """Return whether the active K-builder can use the exact piecewise-linear fast path."""
  if layer_module is None:
    return False

  if isinstance(layer_module, (InitDenseRelu, DenseRelu)):
    return True

  return False


def _stage_uses_linear_controlled_fast_path(stage_spec):
  return getattr(stage_spec, "fast_path_kind", None) == "linear_controlled"


def _stage_uses_piecewise_linear_passive_fast_path(stage_spec):
  return getattr(stage_spec, "fast_path_kind", None) == "piecewise_linear_passive"


def _stage_other_variables(other_model_variables, param_name):
  if param_name is None:
    return {}
  return {key: value.get(param_name, {}) for key, value in other_model_variables.items()}


def lqr_forward_matrices_and_states(batch, params, layers_apply, layer_names, other_model_variables=FrozenDict({})):
  """Build joint first-order transition operators and store the layer states."""
  transitions, transition_transposes = [], []
  states = [batch]
  for i, layer_name in enumerate(layer_names):
    # unravel the layers params so that the jacobians have the right dimension, same for state
    layer_params, unravel_params_fn = ravel_pytree(params[layer_name])
    layer_state = states[i]
    layer_other_vars = {k: v.get(layer_name, {}) for k, v in other_model_variables.items()}

    # Define a simpler state transition function (layer propagation) for jacobians retrieval
    def simpler_apply(parameters, x):
      return layers_apply({'params': unravel_params_fn(parameters)}|layer_other_vars, x, i)

    # Recover next state
    states.append(simpler_apply(layer_params, layer_state))
    transition, transition_transpose = _build_joint_transition_operator(layer_params, layer_state, simpler_apply)
    transitions.append(transition)
    transition_transposes.append(transition_transpose)

  return transitions, transition_transposes, states


def lqr_active_controllable_forward_operators_and_states(batch, params, layers_apply, layer_names,
                                                         other_model_variables=FrozenDict({})):
  """Build active-path operators with a control-only first layer and PyTree-native later state."""
  if not layer_names:
    raise ValueError("lqr_active_controllable_forward_operators_and_states expects at least one layer")

  states = [batch]

  first_layer_name = layer_names[0]
  first_layer_params, unravel_first_params_fn = ravel_pytree(params[first_layer_name])
  first_layer_other_vars = {k: v.get(first_layer_name, {}) for k, v in other_model_variables.items()}
  fixed_input_state = states[0]

  def first_simpler_apply(parameters):
    return layers_apply({'params': unravel_first_params_fn(parameters)} | first_layer_other_vars, fixed_input_state, 0)

  states.append(first_simpler_apply(first_layer_params))
  first_transition, first_transition_transpose = _build_control_only_transition_operator(
    first_layer_params, first_simpler_apply)

  transitions, transition_transposes = [], []
  for i, layer_name in enumerate(layer_names[1:], start=1):
    layer_params, unravel_params_fn = ravel_pytree(params[layer_name])
    layer_state = states[i]
    layer_other_vars = {k: v.get(layer_name, {}) for k, v in other_model_variables.items()}

    def simpler_apply(parameters, x):
      return layers_apply({'params': unravel_params_fn(parameters)} | layer_other_vars, x, i)

    states.append(simpler_apply(layer_params, layer_state))
    transition, transition_transpose = _build_joint_transition_operator(layer_params, layer_state, simpler_apply)
    transitions.append(transition)
    transition_transposes.append(transition_transpose)

  return first_transition, first_transition_transpose, transitions, transition_transposes, states


def lqr_active_execution_forward_operators_and_states(batch, params, stages_apply, execution_stage_specs,
                                                      other_model_variables=FrozenDict({})):
  """Build active-path operators over explicit execution stages, including passive state-only stages."""
  if not execution_stage_specs:
    raise ValueError("lqr_active_execution_forward_operators_and_states expects at least one stage")

  states = [batch]
  stage_operators = []
  seen_control = False

  for execution_index, stage_spec in enumerate(execution_stage_specs):
    stage_state = states[-1]
    stage_other_vars = _stage_other_variables(other_model_variables, stage_spec.param_name)

    if stage_spec.kind == "controlled":
      stage_params, unravel_params_fn = ravel_pytree(params[stage_spec.param_name])

      if not seen_control:
        fixed_input_state = stage_state

        def simpler_apply(parameters):
          return stages_apply({'params': unravel_params_fn(parameters)} | stage_other_vars, fixed_input_state, execution_index)

        states.append(simpler_apply(stage_params))
        forward_op, transpose_op = _build_control_only_transition_operator(stage_params, simpler_apply)
        stage_operators.append(
          ActiveExecutionStageOperator("control_only", stage_spec.name, stage_spec.param_name, forward_op, transpose_op)
        )
        seen_control = True
      else:
        def simpler_apply(parameters, x):
          return stages_apply({'params': unravel_params_fn(parameters)} | stage_other_vars,
                              _cast_state_like(stage_state, x), execution_index)

        states.append(simpler_apply(stage_params, stage_state))
        forward_op, transpose_op = _build_joint_transition_operator(stage_params, stage_state, simpler_apply)
        stage_operators.append(
          ActiveExecutionStageOperator("controlled", stage_spec.name, stage_spec.param_name, forward_op, transpose_op)
        )
    else:
      def simpler_apply(x):
        return stages_apply(stage_other_vars, _cast_state_like(stage_state, x), execution_index)

      states.append(simpler_apply(stage_state))
      forward_op, transpose_op = _build_state_only_transition_operator(stage_state, simpler_apply)
      stage_operators.append(
        ActiveExecutionStageOperator("passive", stage_spec.name, None, forward_op, transpose_op)
      )

  return stage_operators, states


def __lqr_forward_matrices_and_states(batch, params, layers_apply, layer_names, other_model_variables=FrozenDict({})):
  """ Function calculating the A and B matrices (jvp + vjp) of the linear transition layers of the LQR.
  Also store the state variables for each layer application, to use as primal
  """
  a, b, a_transpose, b_transpose = [], [], [], []
  states = [batch]
  for i, layer_name in enumerate(layer_names):
    # unravel the layers params so that the jacobians have the right dimension, same for state
    layer_params, unravel_params_fn = ravel_pytree(params[layer_name])
    layer_state, unravel_state = ravel_pytree(states[i])
    layer_other_vars = {k: v.get(layer_name, {}) for k, v in other_model_variables.items()}

    # Define a simpler state transition function (layer propagation) for jacobians retrieval
    def simpler_apply(parameters, x):
      # return trans_fn.apply({'params': unravel_params_fn(parameters)}, unravel_state(x))
      return layers_apply({'params': unravel_params_fn(parameters)}|layer_other_vars, unravel_state(x), i)

    # Recover next state
    states.append(simpler_apply(layer_params, layer_state))

    # Retrieve the vjp and jvp expressions of the jacobians (w/r to state and to controls)
    def partial_apply_inputs(state):
      return ravel_pytree(simpler_apply(layer_params, state))[0]

    def partial_apply_params(params):
      return ravel_pytree(simpler_apply(params, layer_state))[0]

    # JVPs
    _, b_fn = jax.linearize(partial_apply_params, layer_params)
    _, a_fn = jax.linearize(partial_apply_inputs, jnp.float32(layer_state))
    a.append(a_fn)
    b.append(b_fn)
    # VJPs
    a_transpose_fn = vjp_f(partial_apply_inputs, x=layer_state)
    b_transpose_fn = vjp_f(partial_apply_params, layer_params)
    a_transpose.append(a_transpose_fn)
    b_transpose.append(b_transpose_fn)

  return a, b, a_transpose, b_transpose, states

def _get_cross_entropy_label_smoothing(loss_f):
  if loss_f is cross_entropy_loss:
    return 0.0
  if isinstance(loss_f, Partial) and loss_f.func is cross_entropy_loss:
    if loss_f.args:
      return None
    return loss_f.keywords.get("label_smoothing", 0.0)
  return None


def _is_ngd_divergence(div_f):
  if div_f is ngd_divergence_f:
    return True
  if isinstance(div_f, Partial) and div_f.func is ngd_divergence_f and not div_f.args and not div_f.keywords:
    return True
  return False


def _zero_linear_operator(v):
  return jnp.zeros_like(v)


def _cross_entropy_log_prob_gradient(log_probs, targets, label_smoothing):
  num_classes = log_probs.shape[-1]
  flat_log_probs = log_probs.reshape((-1, num_classes))
  flat_targets = jnp.ravel(targets)
  one_hot = jax.nn.one_hot(flat_targets, num_classes)

  if label_smoothing > 0.0:
    smooth = label_smoothing / num_classes
    one_hot = (1.0 - label_smoothing) * one_hot + smooth

  grad = -one_hot / flat_log_probs.shape[0]
  return grad.reshape(log_probs.shape)


def _lqr_final_costs_and_adjoints_generic(loss_f, final_states, targets, div_f=None, div_arg=None):
  """Generic AD-based terminal-term construction."""
  if div_f: assert div_arg is not None, "div_arg must not be None when a divergence function is specified"
  def loss_fn(outputs):
    return loss_f(outputs, targets)

  grad_fn = jax.grad(loss_fn)
  final_lin_cost = grad_fn(final_states)
  ravel_final_states, unravel_fn = ravel_pytree(final_states)

  if div_f:  # Case where the adjoints are w/r to a divergence function

    def div_fn(outputs):
      return div_f(div_arg, unravel_fn(outputs))
      # return jnp.sum(vmap(div_f)(div_arg, logits))
    grad_div_fn = jax.grad(div_fn)
    final_p = grad_div_fn(final_states)
    _, final_q = jax.linearize(grad_div_fn, jnp.atleast_1d(ravel_final_states))

    return final_q, final_p, final_lin_cost
    # return add_f(Q_T, diag_Ri(1)), p_T, a_T

  else:  # Case where a_T = p_T  -> Newton's method
    _, final_q = jax.linearize(grad_fn, jnp.ravel(jnp.atleast_1d(final_states)))
    # Q_T = diag_Ri(1)  # Should be zero for true gradient descent

    return final_q, final_lin_cost, final_lin_cost


def _lqr_final_costs_and_adjoints_analytic_ce_ngd(final_states, targets, div_arg, label_smoothing):
  final_lin_cost = _cross_entropy_log_prob_gradient(final_states, targets, label_smoothing)
  final_p = -jnp.exp(div_arg)
  return _zero_linear_operator, final_p, final_lin_cost


def _lqr_final_costs_and_adjoints_analytic_ce_newton(final_states, targets, label_smoothing):
  final_lin_cost = _cross_entropy_log_prob_gradient(final_states, targets, label_smoothing)
  return _zero_linear_operator, final_lin_cost, final_lin_cost


def lqr_final_costs_and_adjoints(loss_f, final_states, targets, div_f=None, div_arg=None):
  """Handle terminal linear and quadratic terms for the relaxed LQR objective."""
  label_smoothing = _get_cross_entropy_label_smoothing(loss_f)

  if label_smoothing is not None and div_f is None:
    return _lqr_final_costs_and_adjoints_analytic_ce_newton(final_states, targets, label_smoothing)

  if label_smoothing is not None and _is_ngd_divergence(div_f):
    assert div_arg is not None, "div_arg must not be None when a divergence function is specified"
    return _lqr_final_costs_and_adjoints_analytic_ce_ngd(final_states, targets, div_arg, label_smoothing)

  return _lqr_final_costs_and_adjoints_generic(loss_f, final_states, targets, div_f=div_f, div_arg=div_arg)


def lqr_backward_matrices_and_adjoints(params, states, final_adjoint, transition_transposes, layers_apply, layer_names, damping,
                                       other_model_variables=FrozenDict({})):
  """ Retrieve, in backward order, the Q_i R_i and M_i matrices needed for the resolution of the Riccati equation
  """
  p_backward = [jnp.ravel(jnp.atleast_1d(final_adjoint))]
  q_backward = []
  r_backward = []
  m_backward = []
  m_transpose_backward = []
  for i, layer_name in enumerate(layer_names[::-1]):
    j = len(layer_names) - i - 1  # reverse the index

    # unravel the layers params so that the jacobians have the right dimension, same for state
    layer_params, unravel_fn = ravel_pytree(params[layer_name])
    layer_state, unravel_state = ravel_pytree(states[j])
    layer_other_vars = {k: v.get(layer_name, {}) for k, v in other_model_variables.items()}

    # Define a simpler state transition function (layer propagation) for jacobians retrieval
    def simpler_apply(parameters, x):
      # return trans_fn.apply({'params': unravel_fn(parameters)}, unravel_state(x))
      return layers_apply({'params': unravel_fn(parameters)}|layer_other_vars, unravel_state(x), j)

    def hamiltonian(parameters, p_i, x_i):
      return jnp.dot(ravel_pytree(simpler_apply(parameters, x_i))[0], p_i)

    backward_action = transition_transposes[j](jnp.ravel(p_backward[-1]))
    if isinstance(backward_action, tuple):
      _, state_adjoint = backward_action
    else:
      state_adjoint = backward_action
    p_backward.append(state_adjoint)

    # R and Q calculations can only be removed when using relu activations
    # Get Q matrices
    def hamiltonian_x(x_i):
      return hamiltonian(layer_params, jnp.ravel(p_backward[i]), x_i)

    # _, q_i = jax.linearize(jax.grad(hamiltonian_x, allow_int=True), layer_state)
    def vjp_func(x):
      _, vjp_pullback = jax.vjp(hamiltonian_x, x)
      return vjp_pullback(1.0)[0]  # For scalar-output, this is grad(hamiltonian_x)(x)

    _, q_i = jax.linearize(vjp_func, jnp.float32(layer_state))

    q_backward.append(q_i)

    # Get R matrices
    def hamiltonian_u(parameters):
      return hamiltonian(parameters, jnp.ravel(p_backward[i]), layer_state)

    _, r_i = jax.linearize(jax.grad(hamiltonian_u), layer_params)
    r_i = add_f(r_i, diag_r(damping))  # Replaced by adaptive damping inserted at inversion of R + B^TKB
    r_backward.append(r_i)

    # Get M Matrices
    fn = Partial(jax.grad(hamiltonian), layer_params, p_backward[i])
    _, m_i = jax.linearize(fn, jnp.float32(layer_state))
    m_i_transpose = vjp_f(fn, layer_state)
    m_backward.append(m_i)
    m_transpose_backward.append(m_i_transpose)

  return q_backward, r_backward, m_backward, m_transpose_backward


def lqr_backward_hamiltonian_operators(params, states, final_adjoint, transition_transposes, layers_apply, layer_names,
                                       damping, other_model_variables=FrozenDict({})):
  """Build one joint second-order Hamiltonian operator per layer for the active path."""
  p_backward = [final_adjoint]
  k_backward = []
  for i, layer_name in enumerate(layer_names[::-1]):
    j = len(layer_names) - i - 1

    layer_params, unravel_params_fn = ravel_pytree(params[layer_name])
    layer_state = states[j]
    layer_other_vars = {k: v.get(layer_name, {}) for k, v in other_model_variables.items()}
    layer_state_primal = _state_primal_for_linearization(layer_state)

    def simpler_apply(parameters, x):
      return layers_apply({'params': unravel_params_fn(parameters)} | layer_other_vars, _cast_state_like(layer_state, x), j)

    backward_action = transition_transposes[j](p_backward[-1])
    if isinstance(backward_action, tuple):
      _, state_adjoint = backward_action
    else:
      state_adjoint = backward_action
    p_backward.append(state_adjoint)

    p_i = p_backward[i]

    def hamiltonian_joint(parameters, x):
      return tree_vdot(simpler_apply(parameters, x), p_i)

    _, joint_hessian = jax.linearize(jax.grad(hamiltonian_joint, argnums=(0, 1)), layer_params, layer_state_primal)

    def k_i(control_tangent, state_tangent, joint_hessian=joint_hessian, damping=damping):
      flat_control = jnp.ravel(control_tangent)
      k_u, k_x = joint_hessian(flat_control, state_tangent)
      return k_u + damping * flat_control, k_x

    k_backward.append(k_i)

  return k_backward


def lqr_active_controllable_backward_hamiltonian_operators(params, states, final_adjoint, transition_transposes,
                                                           layers_apply, layer_names, damping,
                                                           other_model_variables=FrozenDict({}),
                                                           layer_modules=None):
  """Build active-path second-order operators with a control-only first layer."""
  if not layer_names:
    raise ValueError("lqr_active_controllable_backward_hamiltonian_operators expects at least one layer")
  if layer_modules is not None and len(layer_modules) != len(layer_names):
    raise ValueError("layer_modules must align with layer_names in the active controllable K builder")

  p_i = final_adjoint
  later_k_backward_rev = []
  for reverse_index, layer_name in enumerate(layer_names[:0:-1]):
    j = len(layer_names) - reverse_index - 1
    layer_params, unravel_params_fn = ravel_pytree(params[layer_name])
    layer_state = states[j]
    layer_other_vars = {k: v.get(layer_name, {}) for k, v in other_model_variables.items()}
    layer_state_primal = _state_primal_for_linearization(layer_state)

    def simpler_apply(parameters, x):
      return layers_apply({'params': unravel_params_fn(parameters)} | layer_other_vars, _cast_state_like(layer_state, x), j)

    def hamiltonian_joint(parameters, x):
      return tree_vdot(simpler_apply(parameters, x), p_i)

    layer_module = None if layer_modules is None else layer_modules[j]
    if _supports_piecewise_linear_active_fast_path(layer_module):
      def grad_params_from_state(x):
        return jax.grad(hamiltonian_joint, argnums=0)(layer_params, x)

      def grad_state_from_params(parameters):
        return jax.grad(hamiltonian_joint, argnums=1)(parameters, layer_state_primal)

      _, mixed_param_from_state = jax.linearize(grad_params_from_state, layer_state_primal)
      _, mixed_state_from_params = jax.linearize(grad_state_from_params, layer_params)

      def k_i(control_tangent, state_tangent, mixed_param_from_state=mixed_param_from_state,
              mixed_state_from_params=mixed_state_from_params, damping=damping):
        flat_control = jnp.ravel(control_tangent)
        return mixed_param_from_state(state_tangent) + damping * flat_control, mixed_state_from_params(flat_control)
    else:
      _, joint_hessian = jax.linearize(jax.grad(hamiltonian_joint, argnums=(0, 1)), layer_params, layer_state_primal)

      def k_i(control_tangent, state_tangent, joint_hessian=joint_hessian, damping=damping):
        flat_control = jnp.ravel(control_tangent)
        k_u, k_x = joint_hessian(flat_control, state_tangent)
        return k_u + damping * flat_control, k_x

    later_k_backward_rev.append(k_i)

    _, p_i = transition_transposes[j - 1](p_i)

  first_layer_name = layer_names[0]
  first_layer_params, unravel_first_params_fn = ravel_pytree(params[first_layer_name])
  first_layer_other_vars = {k: v.get(first_layer_name, {}) for k, v in other_model_variables.items()}
  fixed_input_state = states[0]

  def first_simpler_apply(parameters):
    return layers_apply({'params': unravel_first_params_fn(parameters)} | first_layer_other_vars, fixed_input_state, 0)

  def first_hamiltonian(parameters):
    return tree_vdot(first_simpler_apply(parameters), p_i)

  first_layer_module = None if layer_modules is None else layer_modules[0]
  if _supports_piecewise_linear_active_fast_path(first_layer_module):
    def first_k(control_tangent, damping=damping):
      flat_control = jnp.ravel(control_tangent)
      return damping * flat_control
  else:
    _, first_r = jax.linearize(jax.grad(first_hamiltonian), first_layer_params)

    def first_k(control_tangent, first_r=first_r, damping=damping):
      flat_control = jnp.ravel(control_tangent)
      return first_r(flat_control) + damping * flat_control

  return first_k, list(reversed(later_k_backward_rev))


def lqr_active_execution_backward_hamiltonian_operators(params, states, final_adjoint, execution_stage_operators,
                                                        stages_apply, execution_stage_specs, damping,
                                                        other_model_variables=FrozenDict({}),
                                                        layer_modules=None):
  """Build active-path second-order operators over explicit execution stages."""
  if not execution_stage_specs:
    raise ValueError("lqr_active_execution_backward_hamiltonian_operators expects at least one stage")
  if layer_modules is not None and len(layer_modules) != len(execution_stage_specs):
    raise ValueError("layer_modules must align with execution_stage_specs in the active execution-stage K builder")

  p_i = final_adjoint
  k_rev = []
  for reverse_index, stage_spec in enumerate(reversed(execution_stage_specs)):
    execution_index = len(execution_stage_specs) - reverse_index - 1
    stage_operator = execution_stage_operators[execution_index]
    stage_state = states[execution_index]
    stage_other_vars = _stage_other_variables(other_model_variables, stage_spec.param_name)

    if stage_operator.kind == "passive":
      if _stage_uses_piecewise_linear_passive_fast_path(stage_spec):
        def k_i(state_tangent):
          return zero_tangent_tree_like(state_tangent)
      else:
        layer_state_primal = _state_primal_for_linearization(stage_state)

        def simpler_apply(x):
          return stages_apply(stage_other_vars, _cast_state_like(stage_state, x), execution_index)

        def hamiltonian_x(x):
          return tree_vdot(simpler_apply(x), p_i)

        _, state_hessian = jax.linearize(jax.grad(hamiltonian_x), layer_state_primal)

        def k_i(state_tangent, state_hessian=state_hessian):
          return state_hessian(state_tangent)

      k_rev.append(k_i)
      p_i = stage_operator.transpose(p_i)
      continue

    layer_params, unravel_params_fn = ravel_pytree(params[stage_spec.param_name])
    layer_state_primal = _state_primal_for_linearization(stage_state)

    if stage_operator.kind == "control_only":
      fixed_input_state = stage_state

      if _stage_uses_linear_controlled_fast_path(stage_spec):
        def k_i(control_tangent, damping=damping):
          flat_control = jnp.ravel(control_tangent)
          return damping * flat_control
      else:
        def simpler_apply(parameters):
          return stages_apply({'params': unravel_params_fn(parameters)} | stage_other_vars, fixed_input_state, execution_index)

        def hamiltonian(parameters):
          return tree_vdot(simpler_apply(parameters), p_i)

        _, first_r = jax.linearize(jax.grad(hamiltonian), layer_params)

        def k_i(control_tangent, first_r=first_r, damping=damping):
          flat_control = jnp.ravel(control_tangent)
          return first_r(flat_control) + damping * flat_control

      k_rev.append(k_i)
      continue

    if _stage_uses_linear_controlled_fast_path(stage_spec):
      def k_i(control_tangent, state_tangent, damping=damping):
        flat_control = jnp.ravel(control_tangent)
        return damping * flat_control, zero_tangent_tree_like(state_tangent)
    else:
      def simpler_apply(parameters, x):
        return stages_apply({'params': unravel_params_fn(parameters)} | stage_other_vars,
                            _cast_state_like(stage_state, x), execution_index)

      def hamiltonian_joint(parameters, x):
        return tree_vdot(simpler_apply(parameters, x), p_i)

      _, joint_hessian = jax.linearize(jax.grad(hamiltonian_joint, argnums=(0, 1)), layer_params, layer_state_primal)

      def k_i(control_tangent, state_tangent, joint_hessian=joint_hessian, damping=damping):
        flat_control = jnp.ravel(control_tangent)
        k_u, k_x = joint_hessian(flat_control, state_tangent)
        return k_u + damping * flat_control, k_x

    k_rev.append(k_i)
    _, p_i = stage_operator.transpose(p_i)

  return list(reversed(k_rev))


##################################################
# Preconditioner utils
##################################################
