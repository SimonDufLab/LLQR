""""
A collection of exact 2nd order methods for benchmarking, including the exact solution of LQR by Ricatti equations"""
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree

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
    return s, info  # info == 0 means converged

  return newton_step
