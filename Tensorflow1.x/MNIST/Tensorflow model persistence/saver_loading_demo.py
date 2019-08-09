import tensorflow as tf

# load the graph we already define
saver = tf.train.import_meta_graph("Saved_model/model.ckpt.meta")
with tf.Session() as sess:
    # load variables
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))