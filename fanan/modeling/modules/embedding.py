import flax.linen as nn
import jax.numpy as jnp


class SinusoidalPositionalEmbedding(nn.Module):
    embedding_dim: int
    embedding_min_frequency: float = 1.0
    embedding_max_frequency: float = 10_000.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        half_embedding_dim = self.embedding_dim // 2
        start = jnp.log(self.embedding_min_frequency)
        stop = jnp.log(self.embedding_max_frequency)
        frequencies = jnp.exp(jnp.linspace(start, stop, half_embedding_dim))
        self.angular_speeds = 2.0 * jnp.pi * frequencies

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        positional_embeddings = jnp.concatenate(
            [
                jnp.sin(self.angular_speeds * x),
                jnp.cos(self.angular_speeds * x),
            ],
            axis=-1,
        )
        return positional_embeddings


class TimeEmbedding(nn.Module):
    time_embedding_dim: int
    sinusoidal_embedding_dim: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.positional_embedding = SinusoidalPositionalEmbedding(self.sinusoidal_embedding_dim, dtype=self.dtype)
        self.dense1 = nn.Dense(self.time_embedding_dim, dtype=self.dtype)
        self.dense2 = nn.Dense(self.time_embedding_dim, dtype=self.dtype)

    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        x = self.positional_embedding(time)
        x = self.dense1(x)
        x = nn.gelu(x)
        x = self.dense2(x)
        return x
