import tensorflow as tf

# define two variables and compute the sum of two variables
v1 = tf.Variable(tf.constant(1.0, shape=[1], name='v1'))
v2 = tf.Variable(tf.constant(2.0, shape=[1], name='v2'))
result = v1 + v2

init_op = tf.global_variables_initializer()
#  calim tf.train.Saver to save the model
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    # save model to Saved_model/model.ckpt
    saver.save(sess, "Saved_model/model.ckpt")
