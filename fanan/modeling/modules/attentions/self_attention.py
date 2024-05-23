import math

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from fanan.modeling.modules.attentions.registry import register_attention_fn


@register_attention_fn
@jaxtyped(typechecker=typechecker)
def self_attention(
    query: Float[Array, "batch_size sequence_length n_heads head_dim"],
    value: Float[Array, "batch_size sequence_length n_heads head_dim"],
    key: Float[Array, "batch_size sequence_length n_heads head_dim"],
    mask: jax.Array = None,
) -> Float[Array, "batch_size sequence_length n_heads head_dim"]:
    """Self attention mechanism."""
    kv_heads = key.shape[-2]
    q_heads, head_dim = query.shape[-2], query.shape[-1]

    if q_heads != kv_heads:
        assert q_heads > kv_heads
        tile_factor = q_heads // kv_heads
        key = jnp.repeat(key, tile_factor, axis=-2)
        value = jnp.repeat(value, tile_factor, axis=-2)

    scale = float(1 / math.sqrt(head_dim))

    attention_logits = jnp.einsum("bthd,bThd->bhtT", query, key)
    attention_logits = (attention_logits * scale).astype(query.dtype)

    attention_weights = jax.nn.softmax(attention_logits, axis=-1)
    attention_weights = attention_weights.astype(value.dtype)

    attention_vec = jnp.einsum("bhtT,bThd->bthd", attention_weights, value)

    return attention_vec
