import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # [b, 100] -> [b, 64, 64, 3]
        # [b, 100] -> [b, 3*3*512] -> [b, 3, 3, 512] -> [b, 64, 64, 3]
        self.fc = layers.Dense(3 * 3 * 512)

        self.cl1 = layers.Conv2DTranspose(256, 3, 3, 'valid')
        self.bn1 = layers.BatchNormalization()

        self.cl2 = layers.Conv2DTranspose(128, 5, 2, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.cl3 = layers.Conv2DTranspose(3, 4, 3, 'valid')


    def call(self, inputs, training=None):
        # [b, 100] -> [b, 3*3*512]
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)

        x = self.cl1(x)
        x = self.bn1(x, training)
        x = tf.nn.leaky_relu(x)

        x = self.cl2(x)
        x = self.bn2(x, training)
        x = tf.nn.leaky_relu(x)

        x = self.cl3(x)
        x = tf.tanh(x)

        return x


class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b, 64, 64, 3] -> [b, 1]
        self.cl1 = layers.Conv2D(64, 5, 3, 'valid')

        self.cl2 = layers.Conv2D(128, 5, 3, 'valid')
        self.bn2 = layers.BatchNormalization()

        self.cl3 = layers.Conv2D(256, 5, 3, 'valid')
        self.bn3 = layers.BatchNormalization()

        # [b, h, w, c]-> [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)


    def call(self, inputs, training=None):
        x = self.cl1(inputs)
        x = tf.nn.leaky_relu(x)

        x = self.cl2(x)
        # dont forget the param training
        x = self.bn2(x, training)
        x = tf.nn.leaky_relu(x)

        x = self.cl3(x)
        x = self.bn3(x, training)
        x = tf.nn.leaky_relu(x)

        # [b, h, w, c] -> [b, -1]
        x = self.flatten(x)
        # [b, -1] -> [b, 1]
        logits = self.fc(x)

        return logits


def main():
    d = Discriminator()
    g = Generator()

    # test
    # generate images
    x = tf.random.normal([2, 64, 64, 3])
    z = tf.random.normal([2, 100])

    prob = d(x)
    x_hat = g(z)
    print('output size from generator:', x_hat.shape)

if __name__ == '__main__':
    main()






































