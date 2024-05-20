from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from fanan.modeling.modules.embedding import SinusoidalPositionalEmbedding


class UNetResidualBlock(nn.Module):
    output_channels_width: int
    num_groups: Optional[int] = 8
    dtype: Any = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(self.output_channels_width, kernel_size=(1, 1), name="conv1")
        self.conv2 = nn.Conv(self.output_channels_width, kernel_size=(3, 3), padding="SAME", name="conv2")
        self.conv3 = nn.Conv(self.output_channels_width, kernel_size=(3, 3), padding="SAME", name="conv3")
        self.group_norm = nn.GroupNorm(
            num_groups=self.num_groups,
            epsilon=1e-5,
            use_bias=False,
            use_scale=False,
            dtype=self.dtype,
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        input_width = x.shape[-1]

        residual = self.conv1(x) if input_width != self.output_channels_width else x

        x = self.group_norm(x)
        x = nn.swish(x)
        x = self.conv2(x)
        x = nn.swish(x)
        x = self.conv3(x)

        x = x + residual

        return x


class UNetDownBlock(nn.Module):
    output_channels_width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [
            UNetResidualBlock(
                self.output_channels_width,
                name=f"downblock{i}",
            )
            for i in range(self.block_depth)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.residual_blocks:
            x = block(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x


class UNetUpBlock(nn.Module):
    output_channels_width: int
    block_depth: int

    def setup(self):
        self.residual_blocks = [
            UNetResidualBlock(
                self.output_channels_width,
                name=f"upblock{i}",
            )
            for i in range(self.block_depth)
        ]

    def upsample2d(self, x: jnp.ndarray, scale: int = 2) -> jnp.ndarray:
        batch_size, height, width, channels = x.shape
        upsampled_shape = (batch_size, height * scale, width * scale, channels)
        x = jax.image.resize(x, shape=upsampled_shape, method="bilinear")
        return x

    def __call__(self, x: jnp.ndarray, skip: jnp.ndarray) -> jnp.ndarray:
        x = self.upsample2d(x)
        x = jnp.concatenate([x, skip], axis=-1)
        for block in self.residual_blocks:
            x = block(x)
        return x


class UNet(nn.Module):
    image_size: Tuple[int, int] = (64, 64)
    feature_stages: Tuple[int, ...] = (32, 64, 96, 128)
    block_depth: int = 2
    embedding_dim: int = 64
    embedding_min_frequency: float = 1.0
    embedding_max_frequency: float = 10_000.0

    def setup(self):
        self.sinusoidal_embedding = SinusoidalPositionalEmbedding(
            self.embedding_dim,
            self.embedding_min_frequency,
            self.embedding_max_frequency,
        )
        self.conv1 = nn.Conv(self.feature_stages[0], kernel_size=(1, 1))
        self.down_blocks = [UNetDownBlock(width, self.block_depth) for width in self.feature_stages[:-1]]
        self.residual_blocks = [UNetResidualBlock(self.feature_stages[-1]) for _ in range(self.block_depth)]
        self.up_blocks = [UNetUpBlock(width, self.block_depth) for width in reversed(self.feature_stages[:-1])]

        self.conv2 = nn.Conv(
            3,
            kernel_size=(1, 1),
            kernel_init=nn.initializers.zeros,
        )

    def __call__(
        self,
        noisy_images: jnp.ndarray,
        noise_variances: jnp.ndarray,
    ) -> jnp.ndarray:
        embedding = self.sinusoidal_embedding(noise_variances)
        # TODO: util function for this?
        upsampled_shape = (
            noisy_images.shape[0],
            self.image_size[0],
            self.image_size[1],
            self.embedding_dim,
        )
        embedding = jax.image.resize(embedding, upsampled_shape, method="nearest")

        x = self.conv1(noisy_images)
        x = jnp.concatenate([x, embedding], axis=-1)

        skips = []
        for block in self.down_blocks:
            skips.append(x)
            x = block(x)

        for block in self.residual_blocks:
            x = block(x)

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)

        outputs = self.conv2(x)
        return outputs
