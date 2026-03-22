""" Reimplementation of gpt model in Flax/Jax made with LLM help, based on the implementation in:
https://github.com/fKunstner/class-imbalance-sgd-adam/blob/main/code/src/optexp/models/transformer_encoder.py
"""
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze

from lqr_optimizer._src.utils.utils import EnhancedSequential, StageDescriptor


# -----------------------
# Utilities and primitives
# -----------------------

def causal_attn_mask(seq_len: int, batch_size: int) -> jnp.ndarray:
    """Boolean causal mask [B, L, L] where True = masked (disallowed)."""
    mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1).astype(bool)
    return jnp.broadcast_to(mask, (batch_size, seq_len, seq_len))


class SinusoidalPositionalEncoding(nn.Module):
    dim: int
    dropout: float = 0.0
    max_len: int = 5000
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        """
        x: [B, L, D]
        Adds sine/cosine PE (no params), then applies dropout to embeddings.
        """
        B, L, D = x.shape
        assert D == self.dim, "Embedding dim mismatch for positional encoding."

        # Create PE [L, D]
        pos = jnp.arange(self.max_len)[:L][:, None]  # [L, 1]
        div_term = jnp.exp(jnp.arange(0, D, 2) * (-jnp.log(10000.0) / D))  # [D/2]
        pe = jnp.zeros((L, D), dtype=x.dtype)
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div_term))

        x = x + pe[None, :, :]  # broadcast to [B, L, D]
        x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
        return x


class GPTSelfAttention(nn.Module):
    """Multi-head causal self-attention with output projection and attn dropout."""
    hidden_dim: int
    heads: int
    attn_dim: int  # per-head dim
    attn_dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        """
        x:    [B, L, D]
        returns: [B, L, D]
        """
        B, L, D = x.shape
        h = self.heads
        d = self.attn_dim
        inner = h * d
        scale = d ** -0.5
        mask = causal_attn_mask(L, B)

        qkv = nn.Dense(3 * inner, use_bias=False, name="qkv")(x)  # [B, L, 3*H*d]
        qkv = qkv.reshape(B, L, 3, h, d)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, L, H, d]
        # -> [B, H, L, d]
        q, k, v = (jnp.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))

        # attention scores
        scores = jnp.einsum("b h i d, b h j d -> b h i j", q, k) * scale  # [B,H,L,L]
        # apply causal mask: set masked positions to very negative
        big_neg = jnp.finfo(scores.dtype).min
        scores = jnp.where(mask[:, None, :, :], big_neg, scores)

        # softmax + dropout over last axis (keys)
        attn = nn.softmax(scores, axis=-1)
        attn = nn.Dropout(rate=self.attn_dropout)(attn, deterministic=self.deterministic)

        out = jnp.einsum("b h i j, b h j d -> b h i d", attn, v)  # [B,H,L,d]
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(B, L, inner)  # [B,L,H*d]

        # project back to hidden_dim (skip if perfectly aligned and we want to save a matmul)
        proj = nn.Dense(self.hidden_dim, name="proj")(out) if inner != self.hidden_dim else out
        return proj


class FeedForward(nn.Module):
    dim: int           # input/output dim (hidden_dim)
    mlp_dim: int       # inner width
    resid_dropout: float = 0.0
    approx_gelu: bool = True  # GPT2-style GELU(approx='tanh')
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.mlp_dim, name="fc1")(x)
        if self.approx_gelu:
            y = jax.nn.gelu(y, approximate=True)
        else:
            y = nn.gelu(y)
        y = nn.Dense(self.dim, name="fc2")(y)
        y = nn.Dropout(rate=self.resid_dropout)(y, deterministic=self.deterministic)
        return y


class GPTBlock(nn.Module):
    dim: int
    heads: int
    attn_dim: int
    mlp_dim: Optional[int]  # if None and linear=True in torch, they set 4*dim; we pass the resolved value
    layer_norm: bool = True
    resid_dropout: float = 0.0
    attn_dropout: float = 0.0
    linear: bool = True          # whether to include the FFN block

    layer_norm_eps: float = 1e-5
    norm_first: bool = True      # GPT2 style
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        """Input and output are both the differentiable hidden state `x`."""

        # Self-attention sublayer
        y = x
        if self.norm_first and self.layer_norm:
            y = nn.LayerNorm(epsilon=self.layer_norm_eps)(y)
        y = GPTSelfAttention(
            hidden_dim=self.dim,
            heads=self.heads,
            attn_dim=self.attn_dim,
            attn_dropout=self.attn_dropout,
            deterministic=self.deterministic,
        )(y)
        y = nn.Dropout(rate=self.resid_dropout)(y, deterministic=self.deterministic)
        x = x + y
        if not self.norm_first and self.layer_norm:
            x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        # Feed-forward sublayer (optional)
        if self.linear:
            y2 = x
            if self.norm_first and self.layer_norm:
                y2 = nn.LayerNorm(epsilon=self.layer_norm_eps)(y2)
            y2 = FeedForward(dim=self.dim, mlp_dim=self.mlp_dim, resid_dropout=self.resid_dropout,
                             deterministic=self.deterministic)(
              y2
            )
            x = x + y2
            if not self.norm_first and self.layer_norm:
                x = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)

        return x


class LayerNormCarryStage(nn.Module):
    layer_norm_eps: float = 1e-5
    norm_name: str = "LayerNorm_0"

    @nn.compact
    def __call__(self, x):
        normalized = nn.LayerNorm(epsilon=self.layer_norm_eps, name=self.norm_name)(x)
        return normalized, x


class GPTSelfAttentionNamed(nn.Module):
    hidden_dim: int
    heads: int
    attn_dim: int
    attn_dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        return GPTSelfAttention(
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            attn_dim=self.attn_dim,
            attn_dropout=self.attn_dropout,
            deterministic=self.deterministic,
            name="GPTSelfAttention_0",
        )(x)


class AttentionCoreFromCarryStage(nn.Module):
    hidden_dim: int
    heads: int
    attn_dim: int
    attn_dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs):
        x, residual = inputs
        attended = GPTSelfAttention(
            hidden_dim=self.hidden_dim,
            heads=self.heads,
            attn_dim=self.attn_dim,
            attn_dropout=self.attn_dropout,
            deterministic=self.deterministic,
            name="GPTSelfAttention_0",
        )(x)
        return attended, residual


class ResidualAddDropoutStage(nn.Module):
    rate: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs):
        x, residual = inputs
        x = nn.Dropout(rate=self.rate)(x, deterministic=self.deterministic)
        return residual + x


class FeedForwardFC1Only(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.mlp_dim, name="fc1")(x)


class FeedForwardFC2Only(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.dim, name="fc2")(x)


class FC1FromCarryStage(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(self, inputs):
        x, residual = inputs
        projected = FeedForwardFC1Only(self.mlp_dim, name="FeedForward_0")(x)
        return projected, residual


class GELUFromCarryStage(nn.Module):
    approx_gelu: bool = True

    def __call__(self, inputs):
        x, residual = inputs
        if self.approx_gelu:
            x = jax.nn.gelu(x, approximate=True)
        else:
            x = nn.gelu(x)
        return x, residual


class FC2ResidualStage(nn.Module):
    dim: int
    resid_dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, inputs):
        x, residual = inputs
        projected = FeedForwardFC2Only(self.dim, name="FeedForward_0")(x)
        projected = nn.Dropout(rate=self.resid_dropout)(projected, deterministic=self.deterministic)
        return residual + projected


def _extract_gpt_subtree(source_layer, mapping):
    extracted = {}
    for target_key, source_spec in mapping.items():
        if isinstance(source_spec, dict):
            child_source = source_layer.get(target_key, {})
            extracted[target_key] = _extract_gpt_subtree(child_source, source_spec)
        elif source_spec in source_layer:
            extracted[target_key] = source_layer[source_spec]
    return extracted


def _migrate_gpt_split_tree(loaded_tree, init_tree, legacy_mapping):
    if not init_tree and not loaded_tree:
        return loaded_tree
    if tuple(loaded_tree.keys()) == tuple(init_tree.keys()):
        return loaded_tree

    loaded_keys = list(loaded_tree.keys())
    if len(loaded_keys) != len(legacy_mapping):
        raise ValueError("Legacy GPT checkpoint layer count does not match the expected coarse-stage mapping.")

    migrated = {}
    for old_key, split_targets in zip(loaded_keys, legacy_mapping):
        source_layer = loaded_tree[old_key]
        for new_key, mapping in split_targets:
            if mapping is None:
                migrated[new_key] = source_layer
            else:
                migrated[new_key] = _extract_gpt_subtree(source_layer, mapping)

    ordered = {key: migrated.get(key, init_tree[key]) for key in init_tree.keys()}
    return freeze(ordered)


# -----------------------
# Stack wiring (Init / Final)
# -----------------------

class GPTInitLayer(nn.Module):
    vocab_size: int
    max_length: int
    hidden_dim: int
    heads: int
    attn_dim: int
    mlp_dim: int
    embd_dropout: float = 0.0
    pos_encoding: bool = True
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        """x_in: tokens [B, L] (int32), returns hidden state [B, L, D]."""

        x = x.astype(jnp.int32)
        B, L = x.shape
        if L > self.max_length:
            raise ValueError(f"Sequence too long: {L} > max_length={self.max_length}")

        emb = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim, name="tok_embed")(x)
        x = emb * jnp.sqrt(self.hidden_dim)

        # (Optional) sinusoidal positional encoding + dropout on embeddings
        if self.pos_encoding:
            x = SinusoidalPositionalEncoding(self.hidden_dim, dropout=self.embd_dropout, max_len=self.max_length, deterministic=self.deterministic)(
                x
            )
        else:
            x = nn.Dropout(rate=self.embd_dropout)(x, deterministic=self.deterministic)

        return x


class GPTFinalLayer(nn.Module):
    vocab_size: int
    use_final_ln: bool = True
    layer_norm_eps: float = 1e-5
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        """Returns logits flattened as [B*L, V]."""

        if self.use_final_ln:
            x = nn.LayerNorm(epsilon=self.layer_norm_eps, name="final_ln")(x)
        logits = nn.Dense(self.vocab_size, name="lm_head")(x)  # [B, L, V]
        B, L, V = logits.shape
        logits = logits.reshape((B * L, V))
        logits = nn.log_softmax(logits, axis=-1)

        return logits


# -----------------------
# Factory
# -----------------------

def create_gpt_model(
    num_classes: int, # vocab_size, compatibility with global runner
    emb_dim: int = 768,
    num_heads: int = 12,
    depth: int = 12,
    *,
    width_mlp: Optional[int] = None,     # if None -> 4*emb_dim (GPT-style)
    attn_dropout: float = 0.1,
    resid_dropout: float = 0.1,
    embd_dropout: float = 0.1,
    layer_norm: bool = True,
    pos_encoding: bool = True,
    layer_norm_eps: float = 1e-5,
    max_length: int = 1024,
    attn_dim: Optional[int] = None,      # per-head dim, default emb_dim//num_heads
    linear: bool = True,                 # include FFN
) -> Tuple[EnhancedSequential, EnhancedSequential]:
    """
    Returns:
        (EnhancedSequential, None)
    Notes:
      - Inputs: either tokens [B, L] or a tuple (tokens, deterministic).
      - During training, pass (tokens, False) to enable dropout.
      - Output: logits [B*L, vocab_size], matching Torch model behavior.
    """
    vocab_size = num_classes
    if attn_dim is None:
        if emb_dim % num_heads != 0:
            raise ValueError("emb_dim must be divisible by num_heads when attn_dim is not provided.")
        attn_dim = emb_dim // num_heads

    if width_mlp is None:
        width_mlp = 4 * emb_dim

    def inference_mode(deterministic: bool):
        layers = []
        stage_descriptors = []
        legacy_mapping = []

        def add_controlled(stage_name, module):
            stage_index = len(layers)
            layers.append(module)
            param_name = f"layers_{stage_index}"
            stage_descriptors.append(StageDescriptor(stage_name, "controlled", param_name))
            return param_name

        def add_passive(stage_name, module):
            layers.append(module)
            stage_descriptors.append(StageDescriptor(stage_name, "passive", None))

        legacy_mapping.append(((add_controlled(
            "gpt_init",
            GPTInitLayer(
                vocab_size=vocab_size,
                max_length=max_length,
                hidden_dim=emb_dim,
                heads=num_heads,
                attn_dim=attn_dim,
                mlp_dim=width_mlp,
                embd_dropout=embd_dropout,
                pos_encoding=pos_encoding,
                deterministic=deterministic,
            ),
        ), None),))

        for block_index in range(depth):
            block_mapping = []
            if layer_norm:
                block_mapping.append((add_controlled(
                    f"block_{block_index}_attn_pre_ln",
                    LayerNormCarryStage(layer_norm_eps=layer_norm_eps, norm_name="LayerNorm_0"),
                ), {"LayerNorm_0": "LayerNorm_0"}))
            else:
                raise ValueError("Wave-5.a GPT stage splitting currently requires layer_norm=True.")

            block_mapping.append((add_controlled(
                f"block_{block_index}_attn_core",
                AttentionCoreFromCarryStage(
                    hidden_dim=emb_dim,
                    heads=num_heads,
                    attn_dim=attn_dim,
                    attn_dropout=attn_dropout,
                    deterministic=deterministic,
                ),
            ), {"GPTSelfAttention_0": "GPTSelfAttention_0"}))
            add_passive(f"block_{block_index}_attn_residual", ResidualAddDropoutStage(
                rate=resid_dropout, deterministic=deterministic
            ))

            if linear:
                block_mapping.append((add_controlled(
                    f"block_{block_index}_mlp_pre_ln",
                    LayerNormCarryStage(layer_norm_eps=layer_norm_eps, norm_name="LayerNorm_1"),
                ), {"LayerNorm_1": "LayerNorm_1"}))
                block_mapping.append((add_controlled(
                    f"block_{block_index}_fc1",
                    FC1FromCarryStage(mlp_dim=width_mlp),
                ), {"FeedForward_0": {"fc1": "fc1"}}))
                add_passive(f"block_{block_index}_gelu", GELUFromCarryStage())
                block_mapping.append((add_controlled(
                    f"block_{block_index}_fc2_residual",
                    FC2ResidualStage(dim=emb_dim, resid_dropout=resid_dropout, deterministic=deterministic),
                ), {"FeedForward_0": {"fc2": "fc2"}}))

            legacy_mapping.append(tuple(block_mapping))

        legacy_mapping.append(((add_controlled(
            "gpt_final",
            GPTFinalLayer(
                vocab_size=vocab_size,
                use_final_ln=layer_norm,
                layer_norm_eps=layer_norm_eps,
                deterministic=deterministic,
            ),
        ), None),))

        def migrate_legacy_checkpoint(loaded_params, loaded_batch_stats, init_params, init_batch_stats):
            return (
                _migrate_gpt_split_tree(loaded_params, init_params, legacy_mapping),
                _migrate_gpt_split_tree(loaded_batch_stats, init_batch_stats, legacy_mapping),
            )

        return EnhancedSequential(
            layers,
            stage_descriptors=tuple(stage_descriptors),
            legacy_checkpoint_migrator=migrate_legacy_checkpoint,
        )

    return inference_mode(False), inference_mode(True)
