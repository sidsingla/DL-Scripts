import tensorflow as tf
import os
from tensorflow.python.ops import control_flow_ops

#from deployment import model_deploy
import model_deploy
import matplotlib
import read_records_classification
import preprocess
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pdb

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'

# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', '',
    'Records to use.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')
tf.app.flags.DEFINE_integer('gpu_id', 1,
                            'gpu_id')

tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')

FLAGS = tf.app.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu_id)

num_samples = 1531
# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_name:
        raise ValueError('You must supply the dataset file path with --dataset_name')

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)

        dataset = read_records_classification.input_reader(FLAGS.dataset_name, num_samples)

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        with tf.device(deploy_config.inputs_device()):
            with tf.name_scope('data_provider'):
                provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=FLAGS.num_readers,
                    common_queue_capacity=20 * FLAGS.batch_size,
                    common_queue_min=10 * FLAGS.batch_size,
                    shuffle=True)

            #[image, bbox] = provider.get(['image', 'object/bbox'])
            image, label = provider.get(['image', 'label'])
            image, height, width = preprocess.preprocess_image(image)
        label_fil = open('out/label_fil', 'w')
        with tf.Session() as sess:
            # Initialize all global and local variables
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            # Create a coordinator and run all QueueRunner objects
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # images, keypoints = sess.run([images, keypoints])

            count = 1
            for batch_index in range(15):
                img, label_ = sess.run([image, label])
                img = img.astype(np.float32)
                im_height, im_width, channels = img.shape
                img = np.reshape(img, [im_height, im_width, 3])
                print(im_height, im_width)
                fig = plt.figure(figsize=(20, 20))
                plt.imshow(img)
                fig.savefig('out/test_{}.png'.format(count))
                count = count + 1
                label_fil.write(str(label_) + '\n')

                # print(bbx)
                '''
                bbxs = bbx.tolist()
                for bbx in bbxs:
                    ymin, xmin, ymax, xmax = bbx
                    if math.isnan(xmin):
                        print(bbx)
                        img = img.astype(np.float32)
                        im_height, im_width, channels = img.shape
                        img = np.reshape(img, [im_height, im_width, 3])
                        print(i_heigth, i_width)
                        fig = plt.figure(figsize=(1, 1))
                        plt.imshow(img)
                        fig.savefig('out_bad_images/test_{}.png'.format(count))
                        count = count + 1
                '''
            # Stop the threads
            coord.request_stop()

            # Wait for threads to stop
            coord.join(threads)
            #print(label_fil)
            label_fil.close()
            
if __name__ == '__main__':
    tf.app.run()
#main()
