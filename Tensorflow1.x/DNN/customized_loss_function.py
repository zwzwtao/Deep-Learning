import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# two input nodes
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')

# forward propagation
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# if the prediction is relatively less, the loss will be higher
loss_less = 10
loss_more = 1
# y_ is the correct answer
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_ * loss_more),
                              (y_ - y) * loss_less))
train_step = tf.train.AdamOptimizer(0.01).minimize(loss)

rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)

# here we add a noise to the result of x1 + x1 which is between -0.05 and 00.05
Y = [[x1 + x2 + rdm.rand() / 10 - 0.05] for  (x1, x2) in X]

with tf.Session() as sess:
     init_op = tf.global_variables_initializer()
     sess.run(init_op)
     STEPS = 5000
     for i in range(STEPS):
          start = (i * batch_size) % data_size
          end = min(start + batch_size, data_size)
          sess.run(train_step,
                   feed_dict={x: X[start: end], y_: Y[start: end]})

     print(sess.run(w1))