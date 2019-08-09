import tensorflow as tf

# define a variable to compute exponential moving average
# set as a real number since all the variables requiring computing exponential moing average
# have to be real numbers
v1 = tf.Variable(0, dtype=tf.float32)
# step controls the number of iteration to control decay rate
step = tf.Variable(0, trainable=False)

# define a exponential moving average class, set decay to 0.99, and step to control decay
ema = tf.train.ExponentialMovingAverage(0.99, step)
# update the variable, we provide a list, each time it is implemented, the variables in
# this list will be updated
maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run([v1, ema.average(v1)]))

    # update v1 to 5
    sess.run(tf.assign(v1, 5))
    # compute exponential moving average of m1
    # decay rate is min{0.99, (1 + step) / (10 + step) = 0.1} = 0.1
    # exponential moving average of v1 will be updated as 0.1 * 0 + 0.9 * 5 = 4.5
    sess.run(maintain_average_op)
    # ema.average(v1) gets v1 after doing exponential moving average
    print(sess.run([v1, ema.average(v1)]))

    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    # decay = 0.99
    # 0.99 * 4.5 + 0.01 * 10 = 4.555
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))