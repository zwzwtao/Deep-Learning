import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_OF_CHANNELS = 1
NUM_OF_LABELS = 10

# the depth and size of layer1
CONV1_DEEP = 32    # 32 filters
CONV1_SIZE = 5     # 5 * 5 filter
# the depth and size of layer2
CONV2_DEEP = 64
CONV2_SIZE = 5
# the number of nodes in fully connected layer
FC_SIZE = 512

def inference(input_tensor, train, regularizer):
    '''
    if 'train' set to be true, then it's in trainning precess
    '''
    with tf.variable_scope('layer1-conv1'):
        # the output is a 28 * 28 * 32 matrix
        conv1_weights = conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_OF_CHANNELS, CONV1_DEEP],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        # forward propagation
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # the second layer is max-pooling layer
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2,1], strides=[1,2,2,1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        # output is a 14 * 14 * 64 matrix
        conv2_weithts = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant(0.0))

        conv2 = tf.nn.conv2d(pool1, conv2_weithts, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope('layer4-pool'):
        # output is a 7 * 7 * 64 matrix
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # get the shape of the output from 'layer4-pool' (7 * 7 * 64)
    pool_shape = pool2.get_shape().as_list()

    # convert this matrix to a matrix for a fully connected layer
    # shape[0]: batch size
    # shape[1]: width of image
    # shape[2]: height of image
    # shape[3]: channel of image
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # FC layer
    with tf.variable_scope('layer-5'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        # only FC needs regularization
        if regularizer is not None:
            tf.add_to_collection('loss', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

        # here if it's in training process, use drop-out
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_OF_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('loss', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_OF_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit

