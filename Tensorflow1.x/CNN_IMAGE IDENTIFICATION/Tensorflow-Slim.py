# using TensorFLow raw API to do CNN
with tf.variable_scope(scope_name):
    weights = tf.get_variable("weight", ...)
    biases = tf.get_variable("bias", ...)
    conv = tf.nn.conv2d(...)
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))


# using Tensorflow-Slim, only needs one line
# here there is 3 parameters:
# 1: input matrix
# 2: the depth of current layer
# 3: the size of filter
net = slim.conv2d(input, 32, [3, 3])