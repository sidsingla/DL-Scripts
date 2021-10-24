import os

import tensorflow as tf

slim = tf.contrib.slim


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'object/keypoint/x': 'Keypointx',
    'object/keypoint/y': 'Keypointy',
}

# _NUM_CLASSES = 10

def input_reader(file_path, num_samples):
    keys_to_features = {
        'image/class/label':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/key/sha256':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=1),
        # Object boxes and classes.
        'image/object/bbox/xmin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.VarLenFeature(tf.string),
        'image/object/area':
            tf.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.VarLenFeature(tf.int64),
        'image/object/difficult':
            tf.VarLenFeature(tf.int64),
        'image/object/group_of':
            tf.VarLenFeature(tf.int64),
        'image/object/weight':
            tf.VarLenFeature(tf.float32),
        'image/object/keypoint/x':
            tf.VarLenFeature(tf.float32),
        'image/object/keypoint/y':
            tf.VarLenFeature(tf.float32),
        'image/object/weight/x':
            tf.VarLenFeature(tf.float32),
        'image/object/weight/y':
            tf.VarLenFeature(tf.float32),
        'image/object/face':
            tf.VarLenFeature(tf.float32),
        'image/object/roll':
            tf.VarLenFeature(tf.float32),
        'image/object/pitch':
            tf.VarLenFeature(tf.float32),
        'image/object/yaw':
            tf.VarLenFeature(tf.float32),
        'image/object/pose':
            tf.VarLenFeature(tf.float32),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Image('image/class/label'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/keypoint/x': slim.tfexample_decoder.Tensor('image/object/keypoint/x'),
        'object/keypoint/y': slim.tfexample_decoder.Tensor('image/object/keypoint/y'),
        'object/weight/x': slim.tfexample_decoder.Tensor('image/object/weight/x'),
        'object/weight/y': slim.tfexample_decoder.Tensor('image/object/weight/y'),
        'class': slim.tfexample_decoder.Tensor('image/object/face'),
        'roll': slim.tfexample_decoder.Tensor('image/object/roll'),
        'pitch': slim.tfexample_decoder.Tensor('image/object/pitch'),
        'yaw':  slim.tfexample_decoder.Tensor('image/object/yaw'),
        'pose':  slim.tfexample_decoder.Tensor('image/object/pose'),
        # 'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)


    return slim.dataset.Dataset(
    data_sources=file_path,
    reader=tf.TFRecordReader,
    decoder=decoder,
    num_samples=num_samples,
    items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)
