""" Contain main divergence functions that give rise to specific steepest descent methods like NGD and Newton's descent"""
import jax
import jax.numpy as jnp

# Divergence function (for NGD):
def ngd_divergence_f(px, px_):
  # Taking into account that we return log-softmax and simplifying since we are interested in the second derivative
  # return (-px * jnp.log(px_)).sum() This is when we return softmax
  return (-jnp.exp(px) * px_).sum()