import jax
import flax.linen as nn
import jax.numpy as jnp

from lqr_optimizer._src.utils.utils import EnhancedSequential

class DenseRelu(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.channels)(x)
    return nn.relu(x)

class InitDenseRelu(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))  # Flatten
    return DenseRelu(self.channels)(x)

class DenseLogSoftmax(nn.Module):
  channels: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(self.channels)(x)
    x = nn.log_softmax(x)
    return x

def create_mlp(num_classes: int) -> nn.Module:
  layers = [InitDenseRelu(100),
            DenseRelu(300),
            DenseLogSoftmax(num_classes)]
  return EnhancedSequential(layers)

# class MLP(nn.Module):
#   num_classes: int
#
#   @nn.compact
#   def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#     # Create blocks for EnhancedSequential
#     blocks = [
#       nn.Sequential([nn.Dense(10), nn.relu]),
#       nn.Sequential([nn.Dense(128), nn.relu]),
#       nn.Sequential([nn.Dense(self.num_classes), nn.log_softmax]),  # Returns logits
#     ]
#
#     # Pass the blocks to EnhancedSequential
#     model = EnhancedSequential(blocks)
#     self._layers = model.layers
#
#     # Forward pass through EnhancedSequential
#     x = model(x)
#     return x
#
#   @property
#   def layer_names(self):
#     """Public getter for the stored layer names."""
#     return self._layer_names
#
#   @property
#   def layers(self):
#     """Public getter for the stored layer names."""
#     return self._layers
#
#   def init(self, rng: jax.random.PRNGKey, *args, **kwargs):
#     """
#     Overrides the init method to return the parameter dictionary
#     along with an ordered list of layer names based on the dictionary keys.
#
#     Args:
#         rng: A JAX random key for parameter initialization.
#         *args: Arguments to pass to the forward function.
#         **kwargs: Keyword arguments to pass to the forward function.
#
#     Returns:
#         - params: A FrozenDict containing the initialized parameters.
#         - layer_names: An ordered list of layer names from the params dictionary.
#     """
#     # Call the original init method to get parameters
#     params = super().init(rng, *args, **kwargs)
#
#     # Retrieve layer names directly from the parameter dictionary keys
#     self._layer_names = list(params["params"].keys())
#
#     return params