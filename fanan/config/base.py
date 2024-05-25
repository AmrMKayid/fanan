from datetime import datetime

import yaml
from ml_collections.config_dict import ConfigDict


class MeshConfig(ConfigDict):
    """Mesh class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.data_axis: str = "dp"
        self.fsdp_axis: str = "fsdp"
        self.sequence_axis: str = "sp"
        self.tensor_axis: str = "tp"
        self.n_data_parallel: int = 4
        self.n_fsdp_parallel: int = 2
        self.n_sequence_parallel: int = 2
        self.n_tensors_parallel: int = 1
        self.data_mesh: tuple[str, str] = (self.data_axis, self.fsdp_axis)
        self.sequence_mesh: tuple[str, str] = (self.fsdp_axis, self.sequence_axis)
        self.mesh_axis_names: tuple[str, str, str, str] = (
            self.data_axis,
            self.fsdp_axis,
            self.tensor_axis,
            self.sequence_axis,
        )


class DataConfig(ConfigDict):
    """Data configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.dataset_name: str = "mnist"
        self.image_size: list[int] = [64, 64]
        self.num_channels: int = 3
        self.batch_size: int = 64
        self.cache: bool = False


class DiffusionConfig(ConfigDict):
    """Diffuser configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.timesteps: int = 1000
        self.beta_1: float = 1e-4
        self.beta_t: float = 0.02
        self.timestep_size: float = 0.001
        self.noise_schedule: str = "linear"
        self.ema_decay: float = 0.999
        self.ema_update_every: int = 1
        self.noise_schedule_kwargs: dict = {}
        self.ema_decay_kwargs: dict = {}
        self.diffusion_steps: int = 80


class ArchitectureConfig(ConfigDict):
    """Architecture configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.architecture_name: str = "ddpm"
        self.diffusion: DiffusionConfig = DiffusionConfig()


class LearningRateConfig(ConfigDict):
    """Learning rate configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.schedule_type: str = "constant_warmup"
        self.lr_kwargs: dict = {
            "value": 1e-3,
            "warmup_steps": 128,
        }


class OptimizationConfig(ConfigDict):
    """Optimization configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.optimizer_type: str = "adamw"
        self.optimizer_kwargs: dict = {
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
        }
        self.max_grad_norm: float = 1.0
        self.grad_accum_steps: int = 1
        self.lr_schedule: LearningRateConfig = LearningRateConfig()


class TrainingConfig(ConfigDict):
    """Training configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.total_steps: int = 100_000
        self.eval_every_steps: int = 10
        self.loss_type: str = "l1"
        self.half_precision: bool = True
        self.save_and_sample_every: int = 1000
        self.num_sample: int = 64


class FananConfig(ConfigDict):
    """Fanan configuration class."""

    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.seed: int = 37
        self.timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_every_steps: int = 10


class Config(ConfigDict):
    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.fanan: FananConfig = FananConfig()
        self.mesh: MeshConfig = MeshConfig()
        self.data: DataConfig = DataConfig()
        self.arch: ArchitectureConfig = ArchitectureConfig()
        self.optimization: OptimizationConfig = OptimizationConfig()
        self.training: TrainingConfig = TrainingConfig()

    @classmethod
    def read_config_from_yaml(cls, file_path: str):
        with open(file_path, encoding="utf-8") as file:
            updates = yaml.safe_load(file)

        cfg = cls()
        cfg.update(ConfigDict({**updates}).copy_and_resolve_references())
        return cfg


_mesh_cfg = MeshConfig()

if __name__ == "__main__":
    config = Config()
    print(config)
