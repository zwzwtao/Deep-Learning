import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets
import os
from resnet import resnet_32

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.random.set_seed(2333)


def preprocess(x, y):
    # [-0.5, 0.5]
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    return x, y


(x, y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
train_data_size = x.shape[0]
print(x.shape, y.shape, x_test.shape, y_test.shape)

train_db = tf.data.Dataset.from_tensor_slices((x, y))
train_db = train_db.shuffle(100000).repeat().map(preprocess).batch(512)

test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_db = test_db.map(preprocess).batch(512)

sample_iter = iter(train_db)
sample1 = next(sample_iter)
sample2 = next(sample_iter)
# print(sample1, sample2)
print('Training sample shape:', sample2[0].shape, sample2[1].shape,
      tf.reduce_min(sample2[0]), tf.reduce_max(sample2[0]))


def main():
    # [b, 32, 32, 3] -> [b, 1, 1, 512]
    model = resnet_32()
    model.build(input_shape=(None, 32, 32, 3))
    model.summary()
    optimizer = optimizers.Adam(lr=1e-3)
    # optimizer = optimizers.Adam()
    db_iter = iter(train_db)
    steps_per_epoch = train_data_size // 512

    val_acc = []

    for epoch in range(500):
        # optimizer.learning_rate = 1e-3 * (500 - epoch) / 500
        # for step, (x, y) in enumerate(train_db):
        for step in range(steps_per_epoch):
            with tf.GradientTape() as tape:
                # [b, 32, 32, 3] -> [b, 100]
                training_data, training_label = next(db_iter)
                logits = model(training_data, training=True)
                # [b] -> [b, 100]
                y_onehot = tf.one_hot(training_label, depth=100)
                # compute loss
                loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
                loss = tf.reduce_mean(loss)

                # l2 regularization
                loss_regularization = []
                for p in model.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(tf.stack(loss_regularization))

                loss = loss + 0.0001 * loss_regularization

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 50 == 0:
                print('epoch:', epoch, 'step:', step, 'loss:', float(loss))

        total_num = 0
        total_correct = 0
        for x, y in test_db:
            logits = model(x, training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        val_acc.append(acc)
        print('epoch:', epoch, 'acc:', acc)


if __name__ == '__main__':
    main()
