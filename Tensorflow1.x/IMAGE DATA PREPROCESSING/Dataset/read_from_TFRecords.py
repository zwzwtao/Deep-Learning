import tensorflow as tf


def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    # decode to get pixel matrix
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    image = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'], tf.int32)
    return image, labels


input_files = ["output.tfrecords"]
dataset = tf.data.TFRecordDataset(input_files)
# need map(parser) to parse each data in TFRecords
dataset = dataset.map(parser)

iterator = dataset.make_one_shot_iterator()

image, label = iterator.get_next()

with tf.Session() as sess:
    for i in range(10):
        x, y = sess.run([image, label])
        print(len(x))
