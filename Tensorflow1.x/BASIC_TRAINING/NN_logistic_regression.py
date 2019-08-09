import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# set the first dimension of shape to 'None' in case next time we use a different batch size
# 2 is constant since we don't change the input size...
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# forward propagation
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# back propagation
y = tf.sigmoid(y)

# y_ and y are all vectors
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
dataset_size = 128
# 128*2 vector as input
X = rdm.rand(dataset_size, 2)

# assume if x1 + x2 < 1, then output 1, which means positive sample, else outputs 0
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

# creat a session
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # initialize variables
    sess.run(init_op)

    print("w1, w2 before training: ")
    print(sess.run(w1))
    print(sess.run(w2))

    # let's do 5000 iterations
    STEPS = 5000
    for i in range(STEPS):
        # each time select 'batch_size'(which is 8) samples to train(0 to 8, 8 to 16, 16 to 32 ......)
        # that is to say, we don't train all 128 sample at one time
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        # training, only use 8 samples
        sess.run(train_step,
                 feed_dict={x: X[start: end], y_: Y[start: end]})
        # output the cross_entropy every 1000 iteration
        if i % 1000 == 0:
            # this time we use all the sample to see the cross_entropy so we use x: X, y_: Y
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps, corss entropy on al data is %g" % (i ,total_cross_entropy))

    print("w1, w2 after training: ")
    print(sess.run(w1))
    print(sess.run(w2))



















































# eof
