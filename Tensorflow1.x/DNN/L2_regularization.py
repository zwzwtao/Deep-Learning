w = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w)

# do regularization on parameter w
loss = tf.reduce_mean(tf.square(y_ - y) + tf.contrib.layers.l2_regularizer(lambd)(w))

