import tensorflow as tf


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def crop_and_resize(image: tf.Tensor, resolution: list[int] = [64, 64]) -> tf.Tensor:
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
        size=resolution,
        antialias=True,
        method=tf.image.ResizeMethod.BICUBIC,
    )
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def process_image(
    data: dict[str, tf.Tensor],
    resolution: list[int],
    input_dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    image = data.get("image")
    image = crop_and_resize(image=image, resolution=resolution)
    # image = normalize_to_neg_one_to_one(image)
    image = tf.image.convert_image_dtype(image, input_dtype)
    return image
