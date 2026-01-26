""" Contain main divergence functions that give rise to specific steepest descent methods like NGD and Newton's descent"""
import jax
import jax.numpy as jnp

# Divergence function (for NGD):
def ngd_divergence_f(px, px_):
  # Taking into account that we return log-softmax and simplifying since we are interested in the second derivative
  # return (-px * jnp.log(px_)).sum() This is when we return softmax
  return (-jnp.exp(px) * px_).sum()

def renyi_divergence(px, px_, order=1/2):
  # return (1/order-1) * jnp.log((jnp.exp(px)**order / (jnp.exp(px_)**(order-1))+1e-8).sum())
  # More stable under exponentiation
  return (1/order-1) * jnp.log(((jnp.exp(px)/(jnp.exp(px_) + 1e-12))**(order-1) * jnp.exp(px)).sum())
  # return (1 / order - 1) * jnp.log(((jnp.exp(px) / (jnp.exp(px_))) ** (order - 1) * jnp.exp(px)).sum())

def negative_renyi_divergence(px, px_, order=1/2):
  return -(1/order-1) * jnp.log(((jnp.exp(px)/(jnp.exp(px_) + 1e-12))**(order-1) * jnp.exp(px)).sum())

# Special case of renyi when order tend to infinity
def renyi_inf(px, px_):
  return jnp.log(jnp.max(jnp.exp(px)/(jnp.exp(px_) + 1e-12)))

# Special case of renyi when order tend to 0
def renyi_zero(px, px_):
  return -px_.sum() # Assuming px is bigger than zero everywhere

# Reverse-KL (why not?)
def reverse_kl(px, px_):
  return (jnp.exp(px_) * (px_-px)).sum()