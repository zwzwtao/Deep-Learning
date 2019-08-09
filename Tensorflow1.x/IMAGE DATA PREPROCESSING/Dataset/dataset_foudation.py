import tempfile
import tensorflow as tf

input_data = [1, 2, 3, 5, 7]
dataset = tf.data.Dataset.from_tensor_slices(input_data)

# define iterator
iterator = dataset.make_one_shot_iterator()

# get_next() returns a tensor representing an input data
x = iterator.get_next()
y = x * x

with tf.Session() as sess:
    for i in range(len(input_data)):
        print(sess.run(y))




























