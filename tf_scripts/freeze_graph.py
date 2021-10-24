"""Generic freezing script that freezes a model using the given configurations."""
import pdb
import logging
import os
import tempfile
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.training import saver as saver_lib

slim = tf.contrib.slim

DATA_FORMAT = 'NHWC'
IMAGE_SIZE = 256
use_moving_averages = False

tf.app.flags.DEFINE_string(
    'train_dir', '.',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string(
    'ckpt_name', None,
    'Name of the checkpoint to freeze.')
tf.app.flags.DEFINE_string(
    'out_dir', 'frozen_model/',
    'Directory where saved graph files are written to.')
tf.app.flags.DEFINE_string(
    'dataset_name', '',
    'Records to use.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('num_classes', 1000,
                            'Number of classes.')
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
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_string(
    'dataset', '',
    'Dataset to use')
tf.app.flags.DEFINE_boolean(
    'multiplier', False,
    'Use Multiplier')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'fixed',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0000001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')


tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', 224, 'Train image size')
tf.app.flags.DEFINE_integer('max_number_of_steps', 8000001,
                            'The maximum number of training steps.')

FLAGS = tf.app.flags.FLAGS
add = FLAGS.add
multiplier = FLAGS.multiplier

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(FLAGS.gpu_id)
arr =[]


def freeze_graph_with_def_protos(
    input_graph_def,
    input_saver_def,
    input_checkpoint,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    clear_devices,
    initializer_nodes,
    variable_names_blacklist=''):
  """Converts all variables in a graph and checkpoint into constants."""
  del restore_op_name, filename_tensor_name  # Unused by updated loading code.

  # 'input_checkpoint' may be a prefix if we're using Saver V2 format
  if not saver_lib.checkpoint_exists(input_checkpoint):
    raise ValueError(
        'Input checkpoint "' + input_checkpoint + '" does not exist!')

  if not output_node_names:
    raise ValueError(
        'You must supply the name of a node to --output_node_names.')

  # Remove all the explicit device specifications for this node. This helps to
  # make the graph more portable.
  if clear_devices:
    for node in input_graph_def.node:
      node.device = ''

  print(output_node_names)

  with tf.Graph().as_default():
    tf.import_graph_def(input_graph_def, name='')
    config = tf.ConfigProto(graph_options=tf.GraphOptions())
    with session.Session(config=config) as sess:
      if input_saver_def:
        saver = saver_lib.Saver(saver_def=input_saver_def)
        saver.restore(sess, input_checkpoint)
      else:
        var_list = {}
        reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
          try:
            tensor = sess.graph.get_tensor_by_name(key + ':0')
          except KeyError:
            # This tensor doesn't exist in the graph (for example it's
            # 'global_step' or a similar housekeeping element) so skip it.
            continue
          var_list[key] = tensor
        saver = saver_lib.Saver(var_list=var_list)
        saver.restore(sess, input_checkpoint)
        if initializer_nodes:
          sess.run(initializer_nodes)

      variable_names_blacklist = (variable_names_blacklist.split(',') if
                                  variable_names_blacklist else None)
      output_graph_def = graph_util.convert_variables_to_constants(
          sess,
          input_graph_def,
          output_node_names.split(','),
          variable_names_blacklist=variable_names_blacklist)

  return output_graph_def

def replace_variable_values_with_moving_averages(graph,
                                                 current_checkpoint_file,
                                                 new_checkpoint_file):
  """Replaces variable values in the checkpoint with their moving averages.

  If the current checkpoint has shadow variables maintaining moving averages of
  the variables defined in the graph, this function generates a new checkpoint
  where the variables contain the values of their moving averages.

  Args:
    graph: a tf.Graph object.
    current_checkpoint_file: a checkpoint containing both original variables and
      their moving averages.
    new_checkpoint_file: file path to write a new checkpoint.
  """
  with graph.as_default():
    variable_averages = tf.train.ExponentialMovingAverage(0.0)
    ema_variables_to_restore = variable_averages.variables_to_restore()
    with tf.Session() as sess:
      read_saver = tf.train.Saver(ema_variables_to_restore)
      read_saver.restore(sess, current_checkpoint_file)
      write_saver = tf.train.Saver()
      write_saver.save(sess, new_checkpoint_file)


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
  """Writes the graph and the checkpoint into disk."""
  for node in inference_graph_def.node:
    node.device = ''
  with tf.Graph().as_default():
    tf.import_graph_def(inference_graph_def, name='')
    with session.Session() as sess:
      saver = saver_lib.Saver(saver_def=input_saver_def,
                              save_relative_paths=True)
      saver.restore(sess, trained_checkpoint_prefix)
      saver.save(sess, model_path)


def write_frozen_graph(frozen_graph_path, frozen_graph_def):
  """Writes frozen graph to disk.

  Args:
    frozen_graph_path: Path to write inference graph.
    frozen_graph_def: tf.GraphDef holding frozen graph.
  """
  with gfile.GFile(frozen_graph_path, 'wb') as f:
    f.write(frozen_graph_def.SerializeToString())
  logging.info('%d ops in the final graph.', len(frozen_graph_def.node))

def _image_tensor_input_placeholder(input_shape=None):
  """Returns input placeholder and a 4-D uint8 image tensor."""
  if input_shape is None:
    input_shape = (None, None, None, 3)
  input_tensor = tf.placeholder(
      dtype=tf.uint8, shape=input_shape, name='image_tensor')
  return input_tensor, input_tensor

def _tf_example_input_placeholder():
  """Returns input that accepts a batch of strings with tf examples.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_tf_example_placeholder = tf.placeholder(
      tf.string, shape=[None], name='tf_example')
  def decode(tf_example_string_tensor):
    tensor_dict = slim.TfExampleDecoder().decode(
        tf_example_string_tensor)
    image_tensor = tensor_dict['image']
    return image_tensor
  return (batch_tf_example_placeholder,
          tf.map_fn(decode,
                    elems=batch_tf_example_placeholder,
                    dtype=tf.uint8,
                    parallel_iterations=32,
                    back_prop=False))


def _encoded_image_string_tensor_input_placeholder():
  """Returns input that accepts a batch of PNG or JPEG strings.

  Returns:
    a tuple of input placeholder and the output decoded images.
  """
  batch_image_str_placeholder = tf.placeholder(
      dtype=tf.string,
      shape=[None],
      name='encoded_image_string_tensor')
  def decode(encoded_image_string_tensor):
    image_tensor = tf.image.decode_image(encoded_image_string_tensor,
                                         channels=3)
    image_tensor.set_shape((None, None, 3))
    return image_tensor
  return (batch_image_str_placeholder,
          tf.map_fn(
              decode,
              elems=batch_image_str_placeholder,
              dtype=tf.uint8,
              parallel_iterations=32,
              back_prop=False))

input_placeholder_fn_map = {
    'image_tensor': _image_tensor_input_placeholder,
    'encoded_image_string_tensor':
    _encoded_image_string_tensor_input_placeholder,
    'tf_example': _tf_example_input_placeholder,
}


def _export_inference_graph(
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=None,
                            output_collection_name='inference_op',
                            graph_hook_fn=None):
    tf.gfile.MakeDirs(output_directory)
    frozen_graph_path = os.path.join(output_directory,
                                     'frozen_graph.pb')
    # saved_model_path = os.path.join(output_directory, 'saved_model')
    model_path = os.path.join(output_directory, 'model.ckpt')

    placeholder_args = {}
    if input_shape is not None:
        placeholder_args['input_shape'] = input_shape
    placeholder_tensor, input_tensors = input_placeholder_fn_map['image_tensor'](
        **placeholder_args)

    outputs = clone_fn(input_tensors)

    #slim.get_or_create_global_step()

    saver_kwargs = {}
    if use_moving_averages:
        # This check is to be compatible with both version of SaverDef.
        if os.path.isfile(trained_checkpoint_prefix):
            saver_kwargs['write_version'] = saver_pb2.SaverDef.V1
            temp_checkpoint_prefix = tempfile.NamedTemporaryFile().name
        else:
            temp_checkpoint_prefix = tempfile.mkdtemp()
        replace_variable_values_with_moving_averages(
            tf.get_default_graph(), trained_checkpoint_prefix,
            temp_checkpoint_prefix)
        checkpoint_to_use = temp_checkpoint_prefix
    else:
        checkpoint_to_use = trained_checkpoint_prefix

    saver = tf.train.Saver(**saver_kwargs)
    input_saver_def = saver.as_saver_def()

    write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=checkpoint_to_use)

    if additional_output_tensor_names is not None:
        output_node_names = ','.join(outputs.keys() + additional_output_tensor_names)
    else:
        output_node_names = ','.join(outputs.keys())

    frozen_graph_def = freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=checkpoint_to_use,
        output_node_names=output_node_names,
        restore_op_name='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        initializer_nodes='')
    write_frozen_graph(frozen_graph_path, frozen_graph_def)

# For inception model( Preprocessing will be different for other models )
from inception_preprocessing import preprocess_image
def clone_fn(images):
    images = tf.reshape( images, [IMAGE_SIZE, IMAGE_SIZE, 3] )
    images = preprocess_image( images, IMAGE_SIZE, IMAGE_SIZE, is_training=False )
    images = tf.reshape( images, [FLAGS.batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] )
    
    logits, _ = inference(images, num_classes=36, for_training=False, restore_logits=True, scope=None)

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='classes'),
        'probabilities': tf.nn.softmax(logits, name='probabilities')
    }

    return predictions

# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main():

    # Load mentioned checkpoint or latest checkpoint
    if FLAGS.ckpt_name is not None:
        trained_checkpoint_prefix = FLAGS.ckpt_name
        trained_checkpoint_prefix = os.path.join(FLAGS.train_dir,trained_checkpoint_prefix)
    else:
        checkpoint = tf.train.get_checkpoint_state(FLAGS.train_dir)
        trained_checkpoint_prefix = checkpoint.model_checkpoint_path

    output_directory = FLAGS.out_dir

    _export_inference_graph(
                            use_moving_averages,
                            trained_checkpoint_prefix,
                            output_directory,
                            additional_output_tensor_names=None,
                            input_shape=(FLAGS.batch_size,IMAGE_SIZE,IMAGE_SIZE,3),
                            output_collection_name='inference_op',
                            graph_hook_fn=None)


if __name__ == '__main__':
    main()
