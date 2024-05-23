import flax.linen as nn
import jax
import jax.numpy as jnp


class AttentionKernel(nn.Module):
    mesh: jax.sharding.Mesh
    num_heads: int = 8
    head_dim: int = 64
    attention_kernel_type: str = "dot_product"
    use_memory_efficient_attention: bool = False
    split_head_dim: bool = False
    float32_qk_product: bool = True
    # flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
    flash_min_seq_length: int = 4096
    # flash_block_sizes: BlockSizes = None
    dtype: jnp.dtype = jnp.float32

    def check_attention_inputs(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> None:
        """Check attention inputs."""

        assert key.ndim == value.ndim, "k, v must have same rank."
        assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
        assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
        assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
        assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    def apply_attention(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        decoder_segment_ids: jax.Array | None = None,
    ):
        """Routes to different attention kernels."""
        self.check_attention_inputs(query, key, value)

        scale: float = self.head_dim**-0.5

        can_use_flash_attention = (
            query.shape[1] >= self.flash_min_seq_length
            and key.shape[1] >= self.flash_min_seq_length
            and value.shape[1] >= self.flash_min_seq_length
        )

        if (
            self.attention_kernel_type == "dot_product"
            or self.use_memory_efficient_attention
            or not can_use_flash_attention
        ):
            return self.apply_attention_dot(query, key, value)
        elif self.attention_kernel_type == "flash":
            return self.tpu_flash_attention(query, key * scale, value, decoder_segment_ids)
        else:
            raise ValueError(f"Unexpected attention kernel {self.attention_kernel_type=}.")

    @staticmethod
    def head_split(x: jax.Array, head_dim: int) -> jax.Array:
        return x.reshape(*x.shape[:2], -1, head_dim)  # [batch, seq, emb_dim/head_dim, head_dim]

    def apply_attention_dot(
        self,
        query: jax.Array,  # [batch, seq, emb_dim]
        key: jax.Array,  # [batch, seq, emb_dim]
        value: jax.Array,  # [batch, seq, emb_dim]
    ) -> jax.Array:
        """Apply Attention."""

        scale: float = self.head_dim**-0.5

        query_states = self.head_split(query, self.head_dim)  # [batch, seq, num_heads, head_dim]
        key_states = self.head_split(key, self.head_dim)  # [batch, seq, num_heads, head_dim]
        value_states = self.head_split(value, self.head_dim)  # [batch, seq, num_heads, head_dim]

        if self.float32_qk_product:
            query_states = query_states.astype(jnp.float32)
            key_states = key_states.astype(jnp.float32)

        attention_scores = jnp.einsum("bsnh,bSnh->bnsS", query_states, key_states)  # [batch, num_heads, seq, seq]
        attention_scores = (attention_scores * scale).astype(query_states.dtype)  # [batch, num_heads, seq, seq]

        # TODO: add mask
        mask = None

        attention_probs = jax.nn.softmax(attention_scores, axis=-1)

        attention_probs = attention_probs.astype(self.dtype)

        # attend to values
        hidden_states = jnp.einsum(
            "bnsS,btnh->bsnh", attention_probs, value_states
        )  # [batch, seq, num_heads, head_dim]
        # merge heads
        hidden_states = hidden_states.reshape(hidden_states.shape[:-2] + (-1,))  # [batch, seq, emb_dim]

        return hidden_states


class MultiHeadAttentionBlock(nn.Module):
    # query_dim: int
    num_heads: int = 8
    head_dim: int = 64
    dropout_rate: float = 0.1
    dtype: jnp.dtype = jnp.float32
    attention_kernel_type: str = "dot_product"
    mesh: jax.sharding.Mesh = None

    def setup(self):
        inner_dim: int = self.num_heads * self.head_dim  # equivalent to the embedding_dim

        self.attention_kernel = AttentionKernel(
            mesh=self.mesh,
            attention_kernel_type=self.attention_kernel_type,
            name=f"attention_kernel_{self.attention_kernel_type}",
        )

        qkv_init_kernel = nn.with_logical_partitioning(
            nn.initializers.lecun_normal(),
            ("embed", "heads"),  # TODO: update the partitioning
        )
        self.query_proj = nn.DenseGeneral(
            features=inner_dim,
            kernel_init=qkv_init_kernel,
            use_bias=False,
            dtype=self.dtype,
            name="query_proj",
        )
        self.key_proj = nn.DenseGeneral(
            features=inner_dim,
            kernel_init=qkv_init_kernel,
            use_bias=False,
            dtype=self.dtype,
            name="key_proj",
        )
        self.value_proj = nn.DenseGeneral(
            features=inner_dim,
            kernel_init=qkv_init_kernel,
            use_bias=False,
            dtype=self.dtype,
            name="value_proj",
        )

        self.attention_output_proj = nn.DenseGeneral(
            inner_dim,
            kernel_init=nn.with_logical_partitioning(nn.initializers.lecun_normal(), ("heads", "embed")),
            dtype=self.dtype,
            name="attention_output_proj",
        )

        self.dropout_layer = nn.Dropout(rate=self.dropout_rate, name="dropout")

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        q_proj = self.query_proj(hidden_states)  # (batch, sequence_length, inner_dim)
        k_proj = self.key_proj(context)  # (batch, sequence_length, inner_dim)
        v_proj = self.value_proj(context)  # (batch, sequence_length, inner_dim)

        # query_proj = nn.with_logical_constraint(query_proj, self.query_axis_names)
        # key_proj = nn.with_logical_constraint(key_proj, self.key_axis_names)
        # value_proj = nn.with_logical_constraint(value_proj, self.value_axis_names)

        hidden_states = self.attention_kernel.apply_attention(
            query=q_proj,
            key=k_proj,
            value=v_proj,
        )

        hidden_states = self.attention_output_proj(hidden_states)
        # hidden_states = nn.with_logical_constraint(hidden_states, (BATCH, LENGTH, HEAD))
        hidden_states = self.dropout_layer(hidden_states, deterministic=deterministic)
        return hidden_states
