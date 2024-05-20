from datetime import datetime

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    @classmethod
    def read_config_from_yaml(cls, file_path: str):
        with open(file_path) as file:
            yaml_data = yaml.safe_load(file)

        return cls(**yaml_data)


class MeshConfig(BaseConfig):
    """Mesh class."""

    data_axis: str = "dp"
    fsdp_axis: str = "fsdp"
    sequence_axis: str = "sp"
    tensor_axis: str = "tp"
    n_data_parallel: int = 4
    n_fsdp_parallel: int = 2
    n_sequence_parallel: int = 2
    n_tensors_parallel: int = 1
    data_mesh: tuple[str, str] = (data_axis, fsdp_axis)
    sequence_mesh: tuple[str, str] = (fsdp_axis, sequence_axis)
    mesh_axis_names: tuple[str, str, str, str] = (
        data_axis,
        fsdp_axis,
        tensor_axis,
        sequence_axis,
    )


class DataConfig(BaseConfig):
    """data configuration class."""

    dataset_name: str = "mnist"
    batch_size: int = 64
    cache: bool = False
    image_size: int = 512
    num_channels: int = 3


class DiffusionConfig(BaseConfig):
    """Diffuser configuration class."""

    timesteps: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 0.02
    timestep_size: float = 0.001
    noise_schedule: str = "linear"
    ema_decay: float = 0.999
    ema_update_every: int = 1
    noise_schedule_kwargs: dict = {}
    ema_decay_kwargs: dict = {}
    diffusion_steps: int = 70


class ArchitectureConfig(BaseConfig):
    """Architecture configuration class."""

    architecture_name: str = "ddpm"
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)


class LearningRateConfig(BaseConfig):
    """Learning rate configuration class."""

    schedule_type: str = "constant_warmup"
    lr_kwargs: dict = {
        "value": 1e-3,
        "warmup_steps": 128,
    }


class OptimizationConfig(BaseConfig):
    """Optimization configuration class."""

    optimizer_type: str = "adamw"
    optimizer_kwargs: dict = {
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1e-8,
    }
    max_grad_norm: float = 1.0
    grad_accum_steps: int = 1
    lr_schedule: LearningRateConfig = Field(default_factory=LearningRateConfig)


class TrainingConfig(BaseConfig):
    """training configuration class."""

    total_steps: int = 100_000
    eval_every_steps: int = 10
    loss_type: str = "l1"
    half_precision: bool = True
    save_and_sample_every: int = 1000
    num_sample: int = 64


class FananConfig(BaseConfig):
    """fanan configuration class."""

    seed: int = 37
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_every_steps: int = 10


class Config(BaseConfig):
    fanan: FananConfig = Field(default_factory=FananConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    arch: ArchitectureConfig = Field(default_factory=ArchitectureConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


_mesh_cfg = MeshConfig()

if __name__ == "__main__":
    config = Config()
    print(config)
