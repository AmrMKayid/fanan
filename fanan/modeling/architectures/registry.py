from __future__ import annotations

from typing import Any

from fanan.config import Config
from fanan.modeling.architectures.base import Architecture

_ARCHITECTURES: dict[str, Any] = {}  # registry


def register_architecture(cls):
    global _ARCHITECTURES
    _ARCHITECTURES[cls.__name__.lower()] = cls
    return cls


def get_architecture(config: Config) -> Architecture:
    assert config.arch.architecture_name, "Arch config must specify 'architecture'."
    return _ARCHITECTURES[config.arch.architecture_name.lower()](config)
