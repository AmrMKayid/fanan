from datetime import datetime

import yaml
from pydantic import BaseModel, Field


class BaseConfig(BaseModel):
    """Base configuration class for all configurations."""

    @classmethod
    def read_config_from_yaml(cls, file_path: str) -> "BaseConfig":
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


class FananConfig(BaseConfig):
    """fanan configuration class."""

    PRNGKey: int = 0
    seed: int = 37
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_steps: int = 1000
    log_interval: int = 10


class Config(BaseConfig):
    fanan: FananConfig = Field(default_factory=FananConfig)
    mesh: MeshConfig = Field(default_factory=MeshConfig)


_mesh_cfg = MeshConfig()

if __name__ == "__main__":
    config = Config()
    print(config)
