import tensorflow as tf

input1 = tf.constant([1.0, 2.0], name="input1")
input2 = tf.Variable(tf.random_uniform([2]), name="input2")
input = tf.add_n([input1, input2], name="add")

write = tf.summary.FileWriter("log", tf.get_default_graph())
write.close()
