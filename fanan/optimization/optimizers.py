import optax

from fanan.config import Config


def create_optimizer(
    config: Config,
    lr_schedule: optax.Schedule,
) -> optax.MultiSteps:
    optim_config = config.optimization
    optimizer = None
    match optim_config.optimizer_type:
        case "adam":
            optimizer = optax.adam(learning_rate=lr_schedule, **optim_config.optimizer_kwargs)
        case "adamw":
            optimizer = optax.adamw(learning_rate=lr_schedule, **optim_config.optimizer_kwargs)
        case _:
            raise NotImplementedError(optim_config.optimizer_type)

    optimizer = optax.chain(
        optimizer,
        optax.clip_by_global_norm(max_norm=optim_config.max_grad_norm),
    )
    optimizer = optax.MultiSteps(
        optimizer,
        every_k_schedule=optim_config.grad_accum_steps,
    )
    return optimizer
