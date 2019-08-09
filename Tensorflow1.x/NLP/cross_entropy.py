import tensorflow as tf

word_labels = tf.constant([2, 0])

predict_logits = tf.constant([[2.0, -1, 3.0], [1.0, 0.0, -0.5]])

loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=word_labels, logits=predict_logits)

sess = tf.Session()
print(sess.run(loss))

# softmax_cross_entropy_with_logits is similar, however the predict target should be a distributed
# probability
word_prob_distribution = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=word_prob_distribution, logits=predict_logits)
print(sess.run(loss))


# label smoothing: set the right prob to a number slightly smaller than 1,
# and set the wrong number's prob slightly bigger than 0 to avoid overfit and can improve
# the training result
word_prob_smooth = tf.constant([[0.01, 0.01, 0.98], [0.98, 0.01, 0.01]])
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=word_prob_smooth, logits=predict_logits)
print(sess.run(loss))
sess.close()
