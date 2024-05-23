import unittest

import jax.numpy as jnp

from fanan.modeling.modules.attentions.registry import get_attention_fn


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.sequence_length = 32
        self.n_heads = 8
        self.head_dim = 4

    def test_self_attention(self):
        query = jnp.ones(
            (self.batch_size, self.sequence_length, self.n_heads, self.head_dim),
            dtype=jnp.float32,
        )
        value = jnp.ones(
            (self.batch_size, self.sequence_length, self.n_heads, self.head_dim),
            dtype=jnp.float32,
        )
        key = jnp.ones(
            (self.batch_size, self.sequence_length, self.n_heads, self.head_dim),
            dtype=jnp.float32,
        )

        result = get_attention_fn(name="self_attention")(query, value, key)

        expected_shape = (
            self.batch_size,
            self.sequence_length,
            self.n_heads,
            self.head_dim,
        )
        self.assertEqual(result.shape, expected_shape)

        self.assertTrue(jnp.all(result >= 0))
        self.assertTrue(jnp.all(result <= 1))


class TestAttentionKernels(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.sequence_length = 32
        self.n_heads = 8
        self.head_dim = 16
        self.shape = (
            self.batch_size,
            self.sequence_length,
            self.n_heads,
            self.head_dim,
        )

    def test_attention(self):
        pass
        # rng = jax.random.key(0)
        # key1, key2, key3 = jax.random.split(rng, 3)
        # # query, key, value = (
        # #     jax.random.uniform(key1, self.shape),
        # #     jax.random.uniform(key2, self.shape),
        # #     jax.random.uniform(key3, self.shape),
        # # )
        # hidden_states = jax.random.uniform(key1, (4, 32, 128))
        # flax_mha = nn.MultiHeadDotProductAttention(
        #     num_heads=self.n_heads, qkv_features=self.head_dim
        # )
        # variables = flax_mha.init(jax.random.key(0), hidden_states)
        # flax_out = flax_mha.apply(variables, hidden_states)

        # attn_layer = MultiHeadAttentionBlock(
        #     num_heads=self.n_heads, head_dim=self.head_dim
        # )
        # variables = attn_layer.init(jax.random.key(0), hidden_states)
        # out = attn_layer.apply(variables, hidden_states)

        # import pdb

        # pdb.set_trace()
        # self.assertTrue(
        #     jnp.allclose(out, flax_out), msg=f"{jnp.allclose(out, flax_out)}"
        # )

        # # expected_shape = (
        # #     self.batch_size,
        # #     self.sequence_length,
        # #     self.n_heads,
        # #     self.head_dim,
        # # )
        # # self.assertEqual(output_result.shape, expected_shape)

        # # # self.assertTrue(jnp.allclose(mha_output, output_result))
        # # self.assertTrue(jnp.allclose(out, output_result))

        # # self.assertTrue(jnp.all(output_result >= 0))
        # # self.assertTrue(jnp.all(output_result <= 1))


if __name__ == "__main__":
    unittest.main()
