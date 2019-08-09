import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

reader = tf.TFRecordReader()
# create a queue to maintain input files
filename_queue = tf.train.string_input_producer(["output.tfrecords"])
# read a sample from the file, can also use read_up_to to read multiple samples once at a time
_, serialized_example = reader.read(filename_queue)

# parse one sample, can also use parse_example to parse multiple samples
features = tf.parse_single_example(
    serialized_example,
    features={
        # parse data
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    }
)

# decode_raw() can parse string to relevant pixel array
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
# start thread to process data
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])

