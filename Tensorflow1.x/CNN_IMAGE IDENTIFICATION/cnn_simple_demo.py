# the first two parameters of the 4*1 vector is the size of filter
# the third parameter refers to how many channels each filter have
# the fourth parameter is the number of filters in current layer
filter_weight = tf.get_variable('weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(steddev=0.1))

# all the nodes have the same biases in each layer
# so if there are 16 filters then there is 16 biases
biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

# forward propagation in CNN
# input is a 4*1 vector, the first parameter is the batch size, the last three is the node vector
conv = tf.nn.conv2d(input, filter_weight, strides=[1,1,1,1], padding='SAME')

# the bias can't be added directly, should using the following format
bias = tf.nn.bias_add(conv, biases)
actived_conv = tf.nn.relu(bias)