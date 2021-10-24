import os

import tensorflow as tf

slim = tf.contrib.slim


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'Label of the image.'
}

# _NUM_CLASSES = 10

def input_reader(file_path, num_samples):
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/class/label':
            tf.FixedLenFeature((), tf.int64, default_value=0)
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
    data_sources=file_path,
    reader=tf.TFRecordReader,
    decoder=decoder,
    num_samples=num_samples,
    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
