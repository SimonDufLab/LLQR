import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Union, Callable, Tuple

from lqr_optimizer._src.utils.utils import EnhancedSequential

# --- LayerNorm (scale only, axis=-1) ---
class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    @nn.compact
    def __call__(self, x):
        return nn.LayerNorm(epsilon=self.epsilon, use_scale=True, use_bias=False)(x)

def causal_attn_mask(seq_len: int, batch_size: int) -> jnp.ndarray:
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
    mask = mask == 1
    return jnp.broadcast_to(mask, (batch_size, seq_len, seq_len))

class MaskedAttention(nn.Module):
    hidden_dim: int
    heads: int = 4
    attn_dim: int = 64

    @nn.compact
    def __call__(self, x, mask):
        B, N, _ = x.shape
        dim_head = self.attn_dim
        inner_dim = dim_head * self.heads
        scale = dim_head ** -0.5

        qkv = nn.Dense(inner_dim * 3, use_bias=False)(x)
        qkv = qkv.reshape(B, N, 3, self.heads, dim_head)
        q, k, v = [qkv[:,:,i] for i in range(3)]
        q, k, v = [jnp.transpose(arr, (0, 2, 1, 3)) for arr in (q, k, v)]

        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * scale

        mask = mask[:, None, :, :]
        dots = jnp.where(mask, -1e30, dots)

        attn = nn.softmax(dots, axis=-1)
        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, N, self.heads * dim_head)
        out = nn.Dense(self.hidden_dim)(out) if not (self.heads == 1 and self.attn_dim == self.hidden_dim) else out
        return out

class FeedForward(nn.Module):
    dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x

class TransfLayer(nn.Module):
    dim: int
    heads: int
    attn_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs):
        x, mask = inputs
        x_norm1 = LayerNorm()(x)
        attn_out = MaskedAttention(self.dim, self.heads, self.attn_dim)(x_norm1, mask)
        x = x + attn_out
        x_norm2 = LayerNorm()(x)
        ff_out = FeedForward(self.dim, self.mlp_dim)(x_norm2)
        x = x + ff_out
        return (x, mask)

class InitLayer(nn.Module):
    vocab_size: int
    max_length: int
    heads: int
    hidden_dim: int
    attn_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.int32)
        B, L = x.shape
        assert L <= self.max_length, "sequence too long"
        emb = nn.Embed(self.vocab_size, self.hidden_dim)(x)
        pos_idx = jnp.arange(L)
        pos_emb = self.param("pos_embedding", nn.initializers.normal(stddev=0.02), (self.max_length, self.hidden_dim))
        pos = pos_emb[pos_idx]
        x = emb * jnp.sqrt(self.hidden_dim) + pos[None, :, :]
        mask = causal_attn_mask(L, B)
        inputs = (x, mask)
        x, mask = TransfLayer(self.hidden_dim, self.heads, self.attn_dim, self.mlp_dim)(inputs)
        return (x, mask)

class LastLayer(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, inputs):
        x, mask = inputs
        x = x.reshape((x.shape[0], -1))
        x = LayerNorm()(x)
        x = nn.Dense(self.num_classes)(x)
        x = nn.log_softmax(x)
        return x

def create_grok_model(
    num_classes: int,
    vocab_size: int,
    depth: int = 16,
) -> Tuple[EnhancedSequential, None]:
    mlp_dim = 512
    max_length = 5
    heads = 4
    hidden_dim = 128
    attn_dim = 32

    layers = []
    layers.append(InitLayer(
        vocab_size=vocab_size,
        max_length=max_length,
        heads=heads,
        hidden_dim=hidden_dim,
        attn_dim=attn_dim,
        mlp_dim=mlp_dim,
    ))
    for d in range(1, depth):
        layers.append(TransfLayer(
            dim=hidden_dim,
            heads=heads,
            attn_dim=attn_dim,
            mlp_dim=mlp_dim,
        ))
    layers.append(LastLayer(num_classes=num_classes))

    return EnhancedSequential(layers), None
