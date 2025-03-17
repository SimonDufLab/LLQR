"""Helper functions for building the LQR problem associated with the desired divergence measure"""
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial
from flax.core.frozen_dict import FrozenDict

from lqr_optimizer._src.utils.utils import vjp_f, add_f


def diag_r(penalty):
  return lambda v: penalty * v  # Equivalent to having R_i as an identity matrix x constant


def lqr_forward_matrices_and_states(batch, params, layers_apply, layer_names, other_model_variables=FrozenDict({})):
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
    _, a_fn = jax.linearize(partial_apply_inputs, layer_state)
    a.append(a_fn)
    b.append(b_fn)
    # VJPs
    a_transpose_fn = vjp_f(partial_apply_inputs, x=layer_state)
    # B_T_fn = vjp_f(partial_apply_params, layer_params)
    a_transpose.append(a_transpose_fn)
    # B_T_list.append(B_T_fn)

  return a, b, a_transpose, states

def lqr_final_costs_and_adjoints(loss_f, final_states, targets, div_f=None, div_arg=None):
  """Handle a divergence function, for steepest descent
  """
  if div_f: assert div_arg is not None, "div_arg must not be None when a divergence function is specified"
  # if targets is not None:
  def loss_fn(outputs):
    return loss_f(outputs, targets)
  # else:
  #   loss_fn = lambda v: loss_f(v)

  grad_fn = jax.grad(loss_fn)
  final_lin_cost = grad_fn(final_states)

  if div_f:  # Case where the adjoints are w/r to a divergence function

    def div_fn(outputs):
      return div_f(div_arg, outputs)
      # return jnp.sum(vmap(div_f)(div_arg, logits))
    grad_div_fn = jax.grad(div_fn)
    final_p = grad_div_fn(final_states)
    _, final_q = jax.linearize(grad_div_fn, jnp.atleast_1d(final_states))

    return final_q, final_p, final_lin_cost
    # return add_f(Q_T, diag_Ri(1)), p_T, a_T

  else:  # Case where a_T = p_T  -> Newton's method
    _, final_q = jax.linearize(grad_fn, jnp.atleast_1d(final_states))
    # Q_T = diag_Ri(1)  # Should be zero for true gradient descent

    return final_q, final_lin_cost, final_lin_cost


def lqr_backward_matrices_and_adjoints(params, states, final_adjoint, a_transpose, layers_apply, layer_names, damping,
                                       other_model_variables=FrozenDict({})):
  """ Retrieve, in backward order, the Q_i R_i and M_i matrices needed for the resolution of the Riccati equation
  """
  p_backward = [jnp.atleast_1d(final_adjoint)]
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

    p_backward.append(a_transpose[j](p_backward[-1]))

    # R and Q calculations can only be removed when using relu activations
    # Get Q matrices
    def hamiltonian_x(x_i):
      return hamiltonian(layer_params, p_backward[i], x_i)

    _, q_i = jax.linearize(jax.grad(hamiltonian_x), layer_state)
    q_backward.append(q_i)

    # Get R matrices
    def hamiltonian_u(parameters):
      return hamiltonian(parameters, p_backward[i], layer_state)

    _, r_i = jax.linearize(jax.grad(hamiltonian_u), layer_params)
    r_i = add_f(r_i, diag_r(damping))  # Replaced by adaptive damping inserted at inversion of R + B^TKB
    r_backward.append(r_i)

    # Get M Matrices
    fn = Partial(jax.grad(hamiltonian), layer_params, p_backward[i])
    _, m_i = jax.linearize(fn, layer_state)
    m_i_transpose = vjp_f(fn, layer_state)
    m_backward.append(m_i)
    m_transpose_backward.append(m_i_transpose)

  return q_backward, r_backward, m_backward, m_transpose_backward


##################################################
# Preconditioner utils
##################################################