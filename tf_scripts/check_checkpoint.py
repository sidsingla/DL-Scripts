# Script checks the checkpint model tensors.

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf

latest_ckp = tf.train.latest_checkpoint('dir')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
