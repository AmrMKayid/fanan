from __future__ import annotations

from fanan.config import Config


# class Architecture(ABC, nn.Module):
class Architecture:  # (ABC):
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
