from PIL import Image
import pdb
import numpy as np
import tensorflow as tf

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features = {'image/class/label': tf.FixedLenFeature((), tf.int64, default_value=1),
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=""),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
                'image/format': tf.FixedLenFeature((), tf.string, default_value="jpg")})
    #image = tf.decode_raw(features['image/encoded'], tf.uint8)
    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['image/class/label'], tf.int64)
    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)
    format_ = tf.cast(features['image/format'], tf.string)
    return image, label, height, width, format_


def get_all_records(FILE):
    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer([ FILE ])
        image, label, height, width, format_ = read_and_decode(filename_queue)
        #pdb.set_trace()
        image = tf.reshape(image, tf.stack([height, width, 3]))
        #image.set_shape([256,128,3])
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        i = 0
        for _ in tf.python_io.tf_record_iterator(FILE):
            i += 1
            example, l = sess.run([image, label])
            img = Image.fromarray(example, 'RGB')
            #img.save( "output/" + str(i) + '-train.png')
            #print (l)
        coord.request_stop()
        coord.join(threads)
        print(i)
get_all_records('../mars_tfrecords/train_00000-of-00001.tfrecord')
