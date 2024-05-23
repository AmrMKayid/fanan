from typing import Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from fanan.modeling.modules.embedding import SinusoidalPositionalEmbedding


class UNetResidualBlock(nn.Module):
    output_channels_width: int
    dtype: Any = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(features=self.output_channels_width, kernel_size=(1, 1), name="conv1")
        self.bn = nn.BatchNorm(use_bias=False, use_scale=False)
        self.conv2 = nn.Conv(
            features=self.output_channels_width,
            kernel_size=(3, 3),
            padding="SAME",
            name="conv2",
        )
        self.conv3 = nn.Conv(
            features=self.output_channels_width,
            kernel_size=(3, 3),
            padding="SAME",
            name="conv3",
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:
        input_width = x.shape[3]
        residual = x if input_width == self.output_channels_width else self.conv1(x)

        x = self.bn(
            x,
            use_running_average=not is_training,
        )
        x = self.conv2(x)
        x = nn.swish(x)
        x = self.conv3(x)

        x += residual

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

    def __call__(
        self,
        x: jnp.ndarray,
        skips: list[jnp.ndarray],
        is_training: bool,
    ) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
        for block in self.residual_blocks:
            x = block(x, is_training=is_training)
            skips.append(x)
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

    def __call__(
        self,
        x: jnp.ndarray,
        skips: list[jnp.ndarray],
        is_training: bool,
    ) -> jnp.ndarray:
        x = self.upsample2d(x)
        for block in self.residual_blocks:
            x = jnp.concatenate([x, skips.pop()], axis=-1)
            x = block(x, is_training=is_training)
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
        is_training: bool = True,
    ) -> jnp.ndarray:
        embedding = self.sinusoidal_embedding(noise_variances)
        # TODO: util function for this?
        upsampled_shape = (
            noisy_images.shape[0],
            *self.image_size,
            self.embedding_dim,
        )
        embedding = jax.image.resize(embedding, upsampled_shape, method="nearest")

        x = self.conv1(noisy_images)
        x = jnp.concatenate([x, embedding], axis=-1)

        skips = []
        for block in self.down_blocks:
            x = block(x, skips, is_training)

        for block in self.residual_blocks:
            x = block(x, is_training)

        for block in self.up_blocks:
            x = block(x, skips, is_training)

        outputs = self.conv2(x)
        return outputs
