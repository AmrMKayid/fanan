import logging
from typing import Any

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from fanan.config.base import Config


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def crop_and_resize(image: tf.Tensor, resolution: int = 64) -> tf.Tensor:
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    crop_size = tf.minimum(height, width)
    # image = image[
    #     (height - crop) // 2 : (height + crop) // 2,
    #     (width - crop) // 2 : (width + crop) // 2,
    # ]
    image = tf.image.crop_to_bounding_box(
        image=image,
        offset_height=(height - crop_size) // 2,
        offset_width=(width - crop_size) // 2,
        target_height=crop_size,
        target_width=crop_size,
    )
    image = tf.image.resize(
        image,
        size=(resolution, resolution),
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.cast(image, tf.uint8)


def get_dataset_iterator(config: Config, split: str = "train") -> Any:
    if config.data.batch_size % jax.device_count() > 0:
        raise ValueError(
            f"batch size {config.data.batch_size} must be divisible by the number of devices {jax.device_count()}"
        )

    batch_size = config.data.batch_size // jax.process_count()

    platform = jax.local_devices()[0].platform
    input_dtype = (tf.bfloat16 if platform == "tpu" else tf.float16) if config.training.half_precision else tf.float32

    dataset_builder = tfds.builder(config.data.dataset_name)
    dataset_builder.download_and_prepare()

    def preprocess_fn(d: dict) -> dict[str, Any]:
        image = d.get("image")
        image = crop_and_resize(image=image, resolution=config.data.image_size)
        # image = tf.image.flip_left_right(image)
        image = tf.image.convert_image_dtype(image, input_dtype)
        # return {"image": image}
        return image

    # create split for current process
    num_examples = dataset_builder.info.splits[split].num_examples
    logging.info(f"Total {split=} examples: {num_examples=}")
    split_size = num_examples // jax.process_count()
    logging.info(f"Split size: {split_size=}")
    start = jax.process_index() * split_size
    split = f"{split}[{start}:{start + split_size}]"

    ds = dataset_builder.as_dataset(split=split)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    ds.with_options(options)

    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if config.data.cache:
        ds = ds.cache()

    ds = ds.repeat()
    ds = ds.shuffle(16 * batch_size, seed=config.fanan.seed)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    return iter(tfds.as_numpy(ds))


def get_dataset(config: Config) -> Any:
    train_ds = get_dataset_iterator(config, split="train")
    val_ds = get_dataset_iterator(config, split="test")
    return train_ds, val_ds
