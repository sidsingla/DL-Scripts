# This script is to test tensorflow's frozen graph pb(protobuf) model.

from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf
import random
from PIL import Image, ImageDraw
import time
import pdb
#import sklearn
sys.path.append("..")


flags = tf.app.flags
flags.DEFINE_string('output_dir', 'out', 'Path to the results')
flags.DEFINE_string('output_path', '.', 'Path to output model')
flags.DEFINE_string('test_path', '0', 'Path to test images')
flags.DEFINE_string('gpu_id','7','gpu_id')
FLAGS = flags.FLAGS

TEST_PATH = int(FLAGS.test_path)

PATH_TO_CKPT = '{}/frozen_graph.pb'.format(FLAGS.output_path)

if FLAGS.output_dir not in os.listdir('./'):
    os.mkdir('{}'.format(FLAGS.output_dir))

print('.', end='')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

    #for i in tf.get_default_graph().get_operations():
        #print(i.name)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    
    try:
        im = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    except:
        im = None
    return im

if TEST_PATH==0:
    PATH_TO_TEST_IMAGES_DIR = ''
else:
    raise FileNotFoundError

os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(FLAGS.gpu_id)
img_list = os.listdir(PATH_TO_TEST_IMAGES_DIR)

correct_count = 0
j = 0
class_label_id = {}
fil = open( 'labels.txt', 'r')
mapping = fil.readlines()
labels_list = []
for line in mapping:
    class_id, label = line.split(':')
    labels_list.append( label )
    class_label_id[label.replace('\n', '')] = int(class_id)

labels = []
preds = []
output_fil = open( 'out/output_file', 'w' )
conf_mat_file =  open( 'out/conf_mat_file.txt', 'w' )

prob_dic = {}
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        for imag in img_list:
            imag = imag.replace("\n", "")
            li = imag.split('_')
            class_label = ('_').join(li[:-1])
            class_id = class_label_id[class_label]
            
            image_path = os.path.join(PATH_TO_TEST_IMAGES_DIR, imag)
            image = Image.open(image_path)
            image = image.resize((256,256))
            image_np = load_image_into_numpy_array(image)
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            classes = detection_graph.get_tensor_by_name('classes:0')
            probs = detection_graph.get_tensor_by_name('probabilities:0')

            (out_class, probabilities) = sess.run(
                [classes, probs],
                feed_dict={image_tensor: image_np_expanded})

            out_class = np.squeeze(out_class)
            preds.append( int(out_class) )
            labels.append( class_id )
            for i in range(len(probabilities[0])):
                prob_dic[i] = probabilities[0][i]
                
            sorted_probs = sorted(prob_dic.items(), key=lambda kv: kv[1], reverse=True)
            output_fil.write( str(imag) + ' Correct_Label-> ' + str( class_id ) + ' Predicted_Label-> ' + str( out_class ) + '\n' )
            #print( 'classes', out_class )
            #print( 'Probs', probabilities )

            #image.save('{}/{}'.format(FLAGS.output_dir,imag))
        conf_tensor = tf.confusion_matrix(labels=labels, predictions=preds, num_classes=num_classes, dtype=tf.int32)
        conf_mat_val = sess.run(conf_tensor)
        #conf_mat_file.write( str(conf_mat_val) )
        
import pandas as pd
#pd.DataFrame(conf_mat_val).to_csv("conf_mat.csv")

#print( sklearn.metrics.multilabel_confusion_matrix(y_true=labels, y_pred=preds) )
#print( conf_mat )
#output_fil.close()

import matplotlib
matplotlib.use('Agg')

import seaborn as sns
import matplotlib.pyplot as plt
df_cm = pd.DataFrame(conf_mat_val, index = [i for i in labels_list],
                                       columns = [i for i in labels_list])
plt.figure(figsize = (20,20))
sns.heatmap(df_cm, annot=True)
plt.savefig('conf_mat_sns_2.png')
