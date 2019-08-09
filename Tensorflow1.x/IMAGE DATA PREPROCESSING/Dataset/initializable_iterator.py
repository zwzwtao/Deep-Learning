import tensorflow as tf
from read_from_TFRecords import parser

input_files = tf.placeholder(tf.string)
dataset = tf.data.TFRecordDataset(input_files)
dataset = dataset.map(parser)

# here we need initializable_iterator to go through dataset
iterator = dataset.make_initializable_iterator()
image, label = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer, feed_dict={input_files: ["output.tfrecords"]})

    # if we don't know the range of the data, then use the following loop
    while True:
        try:
            x, y = sess.run([image, label])
            # print(y)
        except tf.errors.OutOfRangeError:
            break
