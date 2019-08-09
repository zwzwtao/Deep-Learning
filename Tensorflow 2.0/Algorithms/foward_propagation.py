import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

# only output error log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x, y), (x_test, y_test) = datasets.mnist.load_data()

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

print(tf.reduce_min(x), tf.reduce_max(x))

x = x / 255
x_test = x_test / 255
y = tf.one_hot(y, depth=10)

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch size:', sample[0].shape, sample[1].shape)

# [batch_size, 784] -> [batch, 256] -> [batch, 128] -> [batch, 10]
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# learning rate: 0.001
lr = 1e-3

for epoch in range(10):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            # o1 = x @ w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            o1 = x @ w1 + b1
            o1 = tf.nn.relu(o1)
            o2 = o1 @ w2 + b2
            o2 = tf.nn.relu(o2)
            output = o2 @ w3 + b3

            # loss
            loss = tf.square(y - output)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, w2, w3, b1, b2, b3])
        w1.assign_sub(lr * grads[0])
        w2.assign_sub(lr * grads[1])
        w3.assign_sub(lr * grads[2])
        b1.assign_sub(lr * grads[3])
        b2.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(step, 'loss', float(loss))

    correct_num, sample_size = 0, 0
    # test
    for step, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 28 * 28])

        o1 = tf.nn.relu(x @ w1 + b1)
        o2 = tf.nn.relu(o1 @ w2 + b2)
        output = o2 @ w3 + b3

        # output shape: [b, 10]
        prob = tf.nn.softmax(output, axis=1)
        # [b, 10] -> [b]
        # dtype of pred is tf.int64 and dtype of y is tf.int32
        pred = tf.argmax(prob, axis=1)

        pred = tf.cast(pred, tf.int32)
        # print('pred type', y.dtype)
        correct = tf.reduce_sum(tf.cast(tf.equal(pred, y), dtype=tf.int32))
        correct_num += correct
        # y.shape: (128,)
        sample_size += y.shape[0]

    acc = correct_num / sample_size
    print("test acc:", acc)
