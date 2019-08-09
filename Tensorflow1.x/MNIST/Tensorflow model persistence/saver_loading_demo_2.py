import tensorflow as tf

v1 = tf.Variable(tf.constant(5.0, shape=[1], name='v1'))
v = tf.Variable(tf.constant(2.0, shape=[1], name='v2'))
result1 = v1 + v

saver = tf.train.Saver()

with tf.Session() as sess:
    # load variables
    saver.restore(sess, "Saved_model/model.ckpt")
    # will output [3.] since the actual value of the variables is from
    # what is saved in model.ckpt
    print(sess.run(result1))