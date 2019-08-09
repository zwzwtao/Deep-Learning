import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# generate integer feature
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# generate string feature
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# convert data to tf.train.Example format
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(np.argmax(label)),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


mnist = input_data.read_data_sets("MNIST_data", dtype=tf.uint8, one_hot=True)
images = mnist.train.images
labels = mnist.train.labels
# images.shape[1] outputs "784"
pixels = images.shape[1]
num_examples = mnist.train.num_examples

# output TFRecord file containing training data
with tf.python_io.TFRecordWriter("output.tfrecords") as writer:
    for index in range(num_examples):
        example = _make_example(pixels, labels[index], images[index])
        writer.write(example.SerializeToString())
print("TFRecord training file has been saved")

# read test data
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# output TFRecord file containing test data
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord testing file has been saved")
