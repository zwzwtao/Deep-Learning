import os
import tensorflow as tf

# ignore some useless log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.mnist.load_data()
train_imgs = tf.convert_to_tensor(train_imgs, dtype=tf.float32) / 255
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
train_label_shape_berfore = train_labels.shape
train_labels = tf.one_hot(train_labels, depth=10)
print("the raw shape of training labels is: {}, after one-hot convert, the shape is: {}"
      .format(train_label_shape_berfore, train_labels.shape))

training_set = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
training_set = training_set.batch(300)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    # attention, for practicing, the activation is not using softmax
    tf.keras.layers.Dense(10)
])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    for step, (train_img, train_label) in enumerate(training_set):
        with tf.GradientTape() as tape:
            train_img = tf.reshape(train_img, (-1, 28 * 28))
            output = model(train_img)
            loss = tf.reduce_sum(tf.square(output - train_label)) / train_img.shape[0]

        # gradient descent
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print("Current epoch: {} , current step: {} , the loss is {} ."
                  .format(epoch, step, loss.numpy()))


def train():
    for epoch in range(30):
        train_epoch(epoch)


if __name__ == '__main__':
    train()
