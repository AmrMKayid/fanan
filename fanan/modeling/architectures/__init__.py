from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import flax.linen as nn
import jax
from jax.sharding import PartitionSpec as PS

from fanan.config import Config

_ARCHITECTURES: dict[str, Any] = {}  # registry


def register_architecture(cls):
    _ARCHITECTURES[cls.__name__.lower()] = cls
    return cls


# class Architecture(ABC, nn.Module):
class Architecture(ABC):
    """Base class for all architectures."""

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def config(self) -> Config:
        return self._config

    # @abstractmethod
    # def __call__(
    #     self, batch: dict[str, jax.Array], training: bool
    # ) -> dict[str, jax.Array]:
    #     pass

    # @abstractmethod
    # def shard(self, ps: PS) -> tuple[Architecture, PS]:
    #     pass


from fanan.modeling.architectures.ddim import *  # isort:skip
# from fanan.modeling.architectures.ddpm import *  # isort:skip


def get_architecture(config: Config) -> Architecture:
    assert config.arch.architecture_name, "Arch config must specify 'architecture'."
    return _ARCHITECTURES[config.arch.architecture_name.lower()](config)
