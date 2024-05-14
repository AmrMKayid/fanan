import unittest

import jax.numpy as jnp

from fanan.modules.attentions import self_attention


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 8
        self.sequence_length = 32
        self.n_heads = 8
        self.head_dim = 4

    def test_self_attention(self):
        query = jnp.ones((self.batch_size, self.sequence_length, self.n_heads, self.head_dim), dtype=jnp.float32)
        value = jnp.ones((self.batch_size, self.sequence_length, self.n_heads, self.head_dim), dtype=jnp.float32)
        key = jnp.ones((self.batch_size, self.sequence_length, self.n_heads, self.head_dim), dtype=jnp.float32)

        result = self_attention(query, value, key)

        expected_shape = (self.batch_size, self.sequence_length, self.n_heads, self.head_dim)
        self.assertEqual(result.shape, expected_shape)

        self.assertTrue(jnp.all(result >= 0))
        self.assertTrue(jnp.all(result <= 1))


if __name__ == "__main__":
    unittest.main()
