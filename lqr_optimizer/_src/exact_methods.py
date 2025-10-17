""""
A collection of exact 2nd order methods for benchmarking, including the exact solution of LQR by Ricatti equations"""
import functools
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from jax.tree_util import Partial

from lqr_optimizer._src.utils.build_lqr import __lqr_forward_matrices_and_states, lqr_final_costs_and_adjoints, lqr_backward_matrices_and_adjoints
from lqr_optimizer._src.utils.utils import add_f, subtract_f

def make_newton_step(loss_to_params, apply_fn, *, damping=1e-4, tol=1e-5, maxiter=None):
  """
  Returns a function newton_step(params, x, y) -> (step, info)
  that computes a damped Newton step s by solving (H + λI) s = g.
  """
  @jax.jit
  def newton_step(params, x, y):
    # Loss closed over (params, batch)
    def loss(p):
      return loss_to_params(p, apply_fn, x, y)

    # Gradient (pytree)
    g = jax.grad(loss)(params)

    # Flatten helpers based on gradient structure (same as params)
    g_flat, unravel = ravel_pytree(g)

    # Define linear operator v -> (H + λI) v, in flat space
    def matvec(v_flat):
      v = unravel(v_flat)
      # HVP via jvp(grad(loss), v)
      hv = jax.jvp(jax.grad(loss), (params,), (v,))[1]
      hv_flat, _ = ravel_pytree(hv)
      return hv_flat + damping * v_flat

    # Solve with CG
    # Returns s_flat such that (H+λI)s = g
    s_flat, info = jsp.sparse.linalg.cg(matvec, g_flat, tol=tol, maxiter=maxiter)

    s = unravel(s_flat)  # back to pytree
    return s  # info == 0 means converged

  return newton_step

def make_lqr_step(divergence_function,
                   loss_fn,
                   model,
                   damping=1e-4,
                   divergence_args_index = -1):
  @jax.jit
  def lqr_step(params, x, y):
    _layer_names = list(params.keys())
    # Retrieve linearization
    a, b, a_transpose, b_transpose, states = __lqr_forward_matrices_and_states(x, params,model.apply_block_from_params,
                                                                _layer_names)
    if divergence_args_index is not None:
      div_arg = states[divergence_args_index]
    else:
      div_arg = None
    final_q, final_p, final_lin_cost = lqr_final_costs_and_adjoints(loss_fn, states[-1], y,
                                                                    div_f=divergence_function,
                                                                    div_arg=div_arg)
    final_lin_cost = jnp.atleast_1d(final_lin_cost)
    q_backward, r_backward, m_backward, m_transpose_backward = lqr_backward_matrices_and_adjoints(params, states,
                                                                                                  final_p,
                                                                                                  a_transpose,
                                                                                                  model.apply_block_from_params,
                                                                                                  _layer_names,
                                                                                                  damping)
    state_sizes = [state.size for state in states]
    lamb_list_rev, BKAM_list_rev, RBKB_inv_list_rev = retrieve_k_lambda_etc(
      final_q, final_lin_cost, a, a_transpose, b, b_transpose, q_backward, r_backward,
      m_backward, m_transpose_backward, state_sizes, _layer_names)

    u_tilde_list = get_update(state_sizes[0], a, b, b_transpose, BKAM_list_rev,
                              lamb_list_rev, RBKB_inv_list_rev, len(_layer_names))
    updates = recast_updates(params, u_tilde_list, _layer_names)

    return updates
  return lqr_step


#############################
# Utils fn for exact LQR retrieval
############################
def scale_operator(operator, scaling_factor):
  return lambda v: scaling_factor*operator(v)

def compose_f(*functions):
  """Performs a composition of the functions passed to it.
  For example, for (f, g, h) given as input, will return lambda v : f(g(h(v)))
  """
  return functools.reduce(lambda f, g: lambda x: f(g(x)), functions,
                          lambda x: x)

def cg_mvp(f):
  """ Return the conjugate gradient algorithm in a form that can be applied directly over a new vector
  """
  return lambda v: jax.scipy.sparse.linalg.cg(f, v, atol=1e-5, maxiter=50)[0]

def store_ki_fn(ki_fn, vector_shape, extra_dim=False):
  """ Develop Ki one column at the time to break recursivity
  """
  xs = jnp.eye(vector_shape)
  if extra_dim:
    xs = jnp.expand_dims(xs, axis=1)
  k_i = jax.vmap(ki_fn, in_axes=0)(xs)
  return k_i

def retrieve_k_lambda_etc(Q_T, a_T, A_list, A_T_list, B_list, B_T_list, Q_list_rev, R_list_rev,
                          M_list_rev, M_T_list_rev, x_sizes, layers_list):
  """ Retrieving all K_i and lambda_i terms, alongside some of the composed terms needed for calculating the update
  """
  Ki_fn_dict = {}
  new_K_dict = {}
  K_list_rev = [Q_T]
  lamb_list_rev = [jnp.atleast_1d(a_T)]
  BKAM_list_rev = []
  BKAM_T_list_rev = []
  RBKB_inv_list_rev = []
  Pow_iter_rev_SV = []
  R_i = []
  for i, trans_fn in enumerate(layers_list[::-1]):
    j = len(layers_list) - i - 1  # reverse the index
    RBKB_i = add_f(
      R_list_rev[i], #add_f(R_i[i], R_list_rev[i]),
      compose_f(B_T_list[j], K_list_rev[i], B_list[j]))
    RBKB_inv_list_rev.append(cg_mvp(RBKB_i))

    if j > 0:
      BKAM_list_rev.append(
        scale_operator(add_f(compose_f(B_T_list[j], K_list_rev[i], A_list[j]),
                             M_list_rev[i]), 1))
      BKAM_T_list_rev.append(
        scale_operator(add_f(M_T_list_rev[i],
                             compose_f(A_T_list[j], K_list_rev[i], B_list[j])), 1))
      lamb_list_rev.append(jnp.atleast_1d(A_T_list[j](lamb_list_rev[i]) -
                           BKAM_T_list_rev[i]
                           (RBKB_inv_list_rev[i]
                            (B_T_list[j](lamb_list_rev[i])))))
      Ki_fn_dict[i] = (subtract_f(
        add_f(compose_f(A_T_list[j], K_list_rev[i], A_list[j]),
              Q_list_rev[i]),
        compose_f(BKAM_T_list_rev[i], RBKB_inv_list_rev[i],
                  BKAM_list_rev[i])))
      # start = time.time()
      if (j+1) % 1 == 0:
        new_K_dict[i] = store_ki_fn(Ki_fn_dict[i], x_sizes[j], extra_dim=False)
        # print(" storing step proceeded in {:.2f} seconds".format(time.time() - start))
        # K_full_matrix_rev.append(new_K)

        def ki(i, v):
          return new_K_dict[i]@v
        K_list_rev.append(Partial(ki, i))
      else:
        def ki(i, v):
          return Ki_fn_dict[i](v)
        K_list_rev.append(Partial(ki, i))

  return lamb_list_rev, BKAM_list_rev, RBKB_inv_list_rev

def get_update(x_0_size, A_list, B_list, B_T_list, BKAM_list_rev, lamb_list_rev,
               RBKB_inv_list_rev, layers_length):

  """Return the solution to the LQR, that is, the parameters update.
  """
  x_tilde_0 = jnp.zeros(x_0_size)
  u_tilde_0 = -RBKB_inv_list_rev[-1](B_T_list[0](lamb_list_rev[-1]))
  x_tilde_list = [x_tilde_0]
  u_tilde_list = [u_tilde_0]
  for i in range(layers_length - 1):
    x_tilde_list.append(A_list[i](x_tilde_list[i]) +
                        B_list[i](u_tilde_list[i]))
    BKAM_x = BKAM_list_rev[-1 - i](x_tilde_list[-1])

    u_tilde_list.append(-RBKB_inv_list_rev[layers_length - 2 - i](BKAM_x + B_T_list[i + 1](lamb_list_rev[layers_length - 2 - i])))

  return u_tilde_list

def recast_updates(params, update_list, layers_name):
  for i, l_name in enumerate(layers_name):
    layer_params, unravel_params_fn = ravel_pytree(params[l_name])
    params[l_name] = unravel_params_fn(-1 * update_list[i])
  return params