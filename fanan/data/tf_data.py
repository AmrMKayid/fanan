from functools import partial
from typing import Any

import jax
import tensorflow as tf
import tensorflow_datasets as tfds
from ml_collections.config_dict import ConfigDict

from fanan.config.base import Config, DataConfig
from fanan.utils.image_utils import process_image


class DefaultDataConfig(DataConfig):
    def __init__(self, initial_dictionary: dict | None = None, **kwargs) -> None:
        super().__init__(initial_dictionary=initial_dictionary, **kwargs)
        self.dataset_name: str = "oxford_flowers102"
        self.image_size: list[int] = [64, 64]
        self.num_channels: int = 3
        self.batch_size: int = 64
        self.cache: bool = False
        self.update(ConfigDict(initial_dictionary).copy_and_resolve_references())


class Dataset:
    def __init__(self, config: Config):
        self._config = config
        self._config.data = DefaultDataConfig(self._config.data)
        self.train_iter, self.val_iter = self.get_dataset()

    def get_dataset(self) -> Any:
        # train_iter = self.get_dataset_iterator(split="train")
        # val_iter = self.get_dataset_iterator(split="test")
        train_iter = self.get_dataset_iterator(split="train[:80%]+validation[:80%]+test[:80%]")
        val_iter = self.get_dataset_iterator(split="train[80%:]+validation[80%:]+test[80%:]")
        return train_iter, val_iter

    def get_dataset_iterator(self, split: str = "train") -> Any:
        if self._config.data.batch_size % jax.device_count() > 0:
            raise ValueError(
                f"batch size {self._config.data.batch_size} must be divisible by the number of devices {jax.device_count()}"
            )

        batch_size = self._config.data.batch_size // jax.process_count()

        platform = jax.local_devices()[0].platform
        input_dtype = (
            (tf.bfloat16 if platform == "tpu" else tf.float16) if self._config.training.half_precision else tf.float32
        )

        ds = tfds.load(self._config.data.dataset_name, split=split, shuffle_files=True)
        ds = ds.map(
            partial(
                process_image,
                resolution=self._config.data.image_size,
                input_dtype=input_dtype,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        if self._config.data.cache:
            ds = ds.cache()

        ds = ds.repeat()
        ds = ds.shuffle(16 * batch_size, seed=self._config.fanan.seed)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return iter(tfds.as_numpy(ds))
