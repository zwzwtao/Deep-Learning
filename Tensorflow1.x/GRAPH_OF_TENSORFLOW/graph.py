import tensorflow as tf

g1 = tf.Graph()
with g1.as_default():
    # define variable 'v' in graph g1, initialize as 0
    v = tf.get_variable(
        "v", shape=[1], initializer=tf.zeros_initializer
    )

g2 = tf.Graph()
with g2.as_default():
    # define variable 'v' in graph g2, initialize as 1
    v = tf.get_variable(
        "v", shape=[1], initializer=tf.ones_initializer
    )

# get variable 'v' in g1
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # outputs [0.]
        print(sess.run(tf.get_variable("v")))

# get variable 'v' in g2
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        # outputs [1.]
        print(sess.run(tf.get_variable("v")))

