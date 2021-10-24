# This script visualises the input image after different tf preprocessing/resizing methods. 

import tensorflow as tf
import Image
import pdb
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
filename = tf.placeholder(tf.string, name="inputFile")
fileContent = tf.read_file(filename, name="loadFile")
image = tf.image.decode_jpeg(fileContent, name="decodeJpeg")
resize_bilinear = tf.image.resize_images(image, size=[256,256], method=tf.image.ResizeMethod.BILINEAR)
resize_bicubic = tf.image.resize_images(image, size=[256,256], method=tf.image.ResizeMethod.BICUBIC)
resize_nearest_neighbor = tf.image.resize_images(image, size=[256,256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
resize_area = tf.image.resize_images(image, size=[256,256], method=tf.image.ResizeMethod.AREA)
resize_image_with_crop_or_pad = tf.image.resize_image_with_crop_or_pad(image, target_height=256, target_width=256)

sess = tf.Session()
feed_dict={filename: "dog_001.jpg"}
with sess.as_default():
    actualImage = image.eval(feed_dict)
    img = Image.fromarray(actualImage)
    img.save("out/actual_img.jpg")

    actual_resize_image_with_crop_or_pad = resize_image_with_crop_or_pad.eval(feed_dict)
    img1 = Image.fromarray(actual_resize_image_with_crop_or_pad)
    img1.save("out/resized_img_with_crop_or_pad.jpg")

    ## Resized images will be distorted if their original aspect ratio is not the same as size.
    actual_resize_bilinear = resize_bilinear.eval(feed_dict)
    img2 = Image.fromarray(actual_resize_bilinear.astype('uint8'))
    img2.save("out/resized_bilinear.jpg")

    actual_resize_nearest_neighbor = resize_nearest_neighbor.eval(feed_dict)
    img3 = Image.fromarray(actual_resize_nearest_neighbor)
    img3.save("out/resized_nearest_neighbor.jpg")

    ## Resized images will be distorted if their original aspect ratio is not the same as size.
    actual_resize_area = resize_area.eval(feed_dict)
    img4 = Image.fromarray(actual_resize_area.astype('uint8'))
    img4.save("out/resized_area.jpg")

    actual_resize_bicubic = resize_bicubic.eval(feed_dict)
    img5 = Image.fromarray(actual_resize_bicubic.astype('uint8'))
    img5.save("out/resized_bicubic.jpg")
