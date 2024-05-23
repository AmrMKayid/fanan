from functools import partial
from typing import Any, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from jax.sharding import PositionalSharding
from ml_collections.config_dict import ConfigDict

from fanan.config.base import ArchitectureConfig, Config
from fanan.modeling.architectures import Architecture, register_architecture
from fanan.modeling.modules.state import TrainState
from fanan.modeling.modules.unet import UNet
from fanan.optimization import lr_schedules, optimizers


class DDIMConfig(ArchitectureConfig):
    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.architecture_name: str = "ddim"

        # UNet parameters
        self.image_size: Tuple[int, int] = (64, 64)
        self.feature_stages: Tuple[int, ...] = (32, 64, 96, 128)
        self.block_depth: int = 4
        self.embedding_min_frequency: float = 1.0
        self.embedding_max_frequency: float = 10_000.0
        self.embedding_dims: int = 64

        # Sampling (reverse diffusion) parameters
        self.min_signal_rate: float = 0.02
        self.max_signal_rate: float = 0.95
        self.update(ConfigDict(initial_dictionary).copy_and_resolve_references())


class DDIMTrainState(TrainState):
    batch_stats: Any
    ema_params: FrozenDict[str, Any]
    ema_step_size: float

    def apply_gradients(self, *, grads, **kwargs):
        next_state = super().apply_gradients(grads=grads, **kwargs)
        new_ema_params = optax.incremental_update(
            new_tensors=next_state.params,
            old_tensors=self.ema_params,
            step_size=self.ema_step_size,
        )
        return next_state.replace(ema_params=new_ema_params)


class DDIMModel(nn.Module):
    config: DDIMConfig

    def setup(self):
        cfg = self.config
        self.normalizer = nn.BatchNorm(use_bias=False, use_scale=False)
        self.network = UNet(
            image_size=cfg.image_size,
            feature_stages=cfg.feature_stages,
            block_depth=cfg.block_depth,
            embedding_dim=cfg.embedding_dims,
            embedding_min_frequency=cfg.embedding_min_frequency,
            embedding_max_frequency=cfg.embedding_max_frequency,
        )

    def __call__(self, images, rng, train: bool):
        images = self.normalizer(images, use_running_average=not train)

        rng_noises, rng_times = jax.random.split(rng)
        noises = jax.random.normal(rng_noises, images.shape, images.dtype)
        diffusion_times = jax.random.uniform(rng_times, (images.shape[0], 1, 1, 1), images.dtype)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, train=train)
        return noises, images, pred_noises, pred_images

    def diffusion_schedule(
        self,
        diffusion_times,
        min_signal_rate: float = 0.02,
        max_signal_rate: float = 0.95,
    ):
        start_angle = jnp.arccos(max_signal_rate)
        end_angle = jnp.arccos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, train: bool):
        pred_noises = self.network(noisy_images, noise_rates**2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        n_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        pred_images = None
        next_noisy_images = initial_noise
        # # # TODO: lax scan?
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate noisy image into noise/image
            ones = jnp.ones((n_images, 1, 1, 1), dtype=initial_noise.dtype)
            diffusion_times = ones - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, train=False)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def denormalize(self, x):
        norm_stats = self.normalizer.variables["batch_stats"]
        mean = norm_stats["mean"].reshape((1, 1, 1, -1)).astype(x.dtype)
        var = norm_stats["var"].reshape((1, 1, 1, -1)).astype(x.dtype)
        std = jnp.sqrt(var + self.normalizer.epsilon)
        return std * x + mean

    def generate(self, rng, image_shape, diffusion_steps: int):
        initial_noise = jax.random.normal(rng, image_shape)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return jnp.clip(generated_images, 0.0, 1.0)


@register_architecture
class DDIM(Architecture):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._config.arch = DDIMConfig(config.arch)
        self.global_step = 0
        self._current_rng = self._init_rng = jax.random.PRNGKey(config.fanan.seed)
        self.sharding = PositionalSharding(jax.devices())
        self._state: DDIMTrainState | None = None
        self._lr_schedule: optax.Schedule | None = None
        if self._state is None:
            self._state, self._lr_schedule = self._create_state()
            self._state = jax.device_put(self._state, self.sharding.replicate())

        self._current_rng, self._rng_train, self._rng_val = jax.random.split(self._current_rng, 3)

    @property
    def initialization_input(self):
        image_size = self.config.data.image_size
        shape = (
            self.config.data.batch_size,
            *image_size,
            self.config.data.num_channels,
        )
        return jnp.ones(shape, dtype=jnp.float32)

    def _create_state(self):
        self._current_rng, key_init, key_diffusion = jax.random.split(self._init_rng, 3)

        model = DDIMModel(config=self._config.arch)
        variables = model.init(
            key_init,
            self.initialization_input,
            key_diffusion,
            train=True,
        )

        tx, lr_schedule = self._create_optimizer()

        state = DDIMTrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            ema_params=variables["params"],
            batch_stats=variables["batch_stats"],
            ema_step_size=self._config.arch.diffusion.ema_update_every,
            tx=tx,
        )

        # Compute FLOPs and Summary
        tabulate_fn = nn.tabulate(
            DDIMModel(config=self._config.arch),
            key_init,
            show_repeated=True,
            compute_flops=True,
            compute_vjp_flops=True,
        )

        print(tabulate_fn(self.initialization_input, key_diffusion, False))

        return state, lr_schedule

    def _create_optimizer(self):
        lr_schedule = lr_schedules.create_lr_schedule(config=self._config)
        optimizer = optimizers.create_optimizer(
            config=self._config,
            lr_schedule=lr_schedule,
        )

        return optimizer, lr_schedule

    def _loss(self, predictions: jnp.ndarray, targets: jnp.ndarray):
        return optax.l2_loss(predictions, targets).mean()  # type:

    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state, batch, rng):
        def loss_fn(params):
            outputs, mutated_vars = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats}, batch, rng, train=True, mutable=["batch_stats"]
            )
            noises, images, pred_noises, pred_images = outputs

            noise_loss = self._loss(pred_noises, noises)
            image_loss = self._loss(pred_images, images)
            loss = noise_loss + image_loss
            return loss, mutated_vars

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, mutated_vars), grads = grad_fn(state.params)
        state = state.apply_gradients(
            grads=grads,
            batch_stats=mutated_vars["batch_stats"],
        )
        return state, loss

    def train_step(self, batch):
        self._rng_train, rng = jax.random.split(self._rng_train)
        new_state, loss = self._train_step(self._state, batch, rng)
        self._state = new_state
        self.global_step += 1
        return loss

    # @partial(jax.jit, static_argnums=(0,5))
    def _eval_step(self, state, params, rng, batch, diffusion_steps: int):
        variables = {"params": params, "batch_stats": state.batch_stats}
        generated_images = state.apply_fn(variables, rng, batch.shape, diffusion_steps, method=DDIMModel.generate)
        return generated_images

    def eval_step(self, batch):
        diffusion_steps = self.config.arch.diffusion.diffusion_steps
        generated_images = self._eval_step(
            state=self._state,
            params=self._state.ema_params,
            rng=self._rng_val,
            batch=batch,
            diffusion_steps=diffusion_steps,
        )
        return generated_images
