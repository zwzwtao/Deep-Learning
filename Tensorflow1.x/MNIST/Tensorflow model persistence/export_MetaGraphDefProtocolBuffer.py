import tensorflow as tf

vo = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = vo + v2

saver = tf.train.Saver()


saver.export_meta_graph("Saved_Model/model.ckpt.meta.json", as_text=True)
