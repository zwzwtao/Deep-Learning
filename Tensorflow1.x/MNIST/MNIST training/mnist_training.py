import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# input nodes, set as 784(pixels)
INPUT_NODE = 784
# 0-9, thus 10 output nodes
OUTPUT_NODE = 10

# neural network parameters
# we have 2 layers, here we set layer1_node, the following layers will be the same size
LAYER1_NODE = 500
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    '''
    given input and all the parameters, output the result of forward propagation
    activation function: ReLU
    moving average can be used
    '''
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # here we don't have to compute the result of softmax since later it will
        # be computed in loss function(see cross_entropy)
        return tf.matmul(layer1, weights2) + biases2

    else:
        # first use avg_class.average to compute moving average of parameters
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

# training steps
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-output')

    # parameters in hidden layer
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # parameters in output layer
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # compute the result of forward propagation, here we don't use mvoing average
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # define a variable that records the number of iteration
    # usually we define a variable that records the number iteration as untrainable
    global_step = tf.Variable(0, trainable=False)

    # initialize moving average class
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # tf.trainable_variables returns all the parameters which don't set trainable as False
    variables_average_op = variable_averages.apply(tf.trainable_variables())

    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # using cross_entropy as the loss function
    # nn.sparse_softmax_cross_entropy_with_logits() takes two parameters:
    #   1. the result of forward propagation without computing softmax
    #   2. the correct result of training samples, since it takes an exact number to
    #      represent the label, so we use argmax to return the index of the array
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # compute the mean of cross_entropy in current batch
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # compute L2 rugularization function
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # compute regularization loss
    regularization = regularizer(weights1) + regularizer(weights2)
    # the total loss
    loss = cross_entropy_mean + regularization
    # set exponential decay rate of learning rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               mnist.train.num_examples / BATCH_SIZE,   # the number of iterations it
                                                                                        # takes to train all the samples
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    # optimizer
    # we assign global_step to blobal_step to automatically increase the variable 'global_step'
    # after each iteration
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # combine two steps
    with tf.control_dependencies([train_step, variables_average_op]):
        train_op = tf.no_op(name='train')

    # compare the result from the neural network and the correct result
    # the type of tf.equal is boolean, thus we have to cast it to real number
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        print(sess.run(tf.argmax(mnist.validation.labels, 1)))
        # test data
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using average model is %g" % (i, validate_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})


mnist = input_data.read_data_sets("mnist/", one_hot=True)
train(mnist)
























































# eof