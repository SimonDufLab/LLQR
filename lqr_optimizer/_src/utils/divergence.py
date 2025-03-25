""" Contain main divergence functions that give rise to specific steepest descent methods like NGD and Newton's descent"""
import jax
import jax.numpy as jnp

# Divergence function (for NGD):
def ngd_divergence_f(px, px_):
  # Taking into account we return log-softmax
  return (-px * jnp.log(px_)).sum()