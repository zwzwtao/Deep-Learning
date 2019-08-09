# current iteration
global_step = tf.Variable(0)

# generate learning rate using exponential decay
# the initial learning rate is 0.1, since staircase is set to True,
# learning_rate will multiply 0.96 every 100 iteration
# here 100 means it takes 100 iterations to go(train) through all the samples
learning_rate = tf.train.expoential_decay(0.1, global_step, 100, 0.96, staircase=True)

# learning using exponential decay, we pass 'global_step' to minimize,
# and global_step will automatically update and so as the learning_rate
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

