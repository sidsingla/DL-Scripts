
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops
import math

IMAGE_RESIZE = 56
HM_SIZE = 10
# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

def bbox(image, bbox):
    def _ImageDimensions(image):
        """Returns the dimensions of an image tensor.
        Args:
          image: A 3-D Tensor of shape `[height, width, channels]`.
        Returns:
          A list of `[height, width, channels]` corresponding to the dimensions of the
            input image.  Dimensions that are statically known are python integers,
            otherwise they are integer scalar tensors.
        """
        if image.get_shape().is_fully_defined():
            return image.get_shape().as_list()
        else:
            static_shape = image.get_shape().with_rank(3).as_list()
            dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]

    def resize_image(image, size,
                     method=tf.image.ResizeMethod.BILINEAR,
                     align_corners=False):
        """Resize an image and bounding boxes.
        """
        # Resize image.
        with tf.name_scope('resize_image'):
            height, width, channels = _ImageDimensions(image)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, size,
                                           method, align_corners)
            image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
            return image, tf.convert_to_tensor(height, dtype=tf.int32), tf.convert_to_tensor(width, dtype=tf.int32)

    with tf.name_scope('Preprocessing', [image, bbox]):

        # Resize image to output size.
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image, height, width = resize_image(image, [300, 300],
                                            method=tf.image.ResizeMethod.BILINEAR,
                                            align_corners=False)

        length = bbox.get_shape().as_list()

        shape = tf.shape(bbox)

        print(length)

        bbox = tf.concat(bbox, axis=1)
        print(bbox)

        bbox = tf.reshape(bbox, shape)
        return image, bbox, height, width


def preprocess_image(image):
    def _ImageDimensions(image):
        """Returns the dimensions of an image tensor.
        Args:
          image: A 3-D Tensor of shape `[height, width, channels]`.
        Returns:
          A list of `[height, width, channels]` corresponding to the dimensions of the
            input image.  Dimensions that are statically known are python integers,
            otherwise they are integer scalar tensors.
        """
        if image.get_shape().is_fully_defined():
            return image.get_shape().as_list()
        else:
            static_shape = image.get_shape().with_rank(3).as_list()
            dynamic_shape = array_ops.unstack(array_ops.shape(image), 3)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]

    def resize_image(image, size,
                     method=tf.image.ResizeMethod.BILINEAR,
                     align_corners=False):
        """Resize an image and bounding boxes.
        """
        # Resize image.
        with tf.name_scope('resize_image'):
            height, width, channels = _ImageDimensions(image)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_images(image, size,
                                           method, align_corners)
            image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
            return image, tf.convert_to_tensor(height, dtype=tf.int32), tf.convert_to_tensor(width, dtype=tf.int32)

    with tf.name_scope('Preprocessing', image):
        # Resize image to output size.
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image, height, width = resize_image(image, [256, 256],
                                            method=tf.image.ResizeMethod.BILINEAR,
                                            align_corners=False)

        return image, height, width
