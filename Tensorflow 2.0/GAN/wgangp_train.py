import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import glob
from wgan import Generator, Discriminator
from dataset import make_anime_dataset


def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        return img

    preprocessed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocessed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocessed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b + 1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single_row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)


def cross_entropy_loss_ones(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,  # shape of logits: [b, 1]
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def cross_entropy_loss_zeros(logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,  # shape of logits: [b, 1]
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)


def gradient_penalty(discriminator, real_img, fake_img):
    batch_size = real_img.shape[0]
    # add
    t = tf.random.uniform([batch_size, 1, 1, 1])
    t = tf.broadcast_to(t, fake_img.shape)
    # shape of interpolate: [batch_size_real_img/batch_size_fake_img(both are the same), h, w, c]
    interpolate = t * real_img + (1 - t) * fake_img

    with tf.GradientTape() as tape:
        # interpolate is not a variable
        tape.watch([interpolate])
        d_interpolate_logits = discriminator(interpolate)
    # grads shape: [b, h, w, c]
    grads = tape.gradient(d_interpolate_logits, interpolate)

    # flatten: [b, h, w, c] -> [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    # shape: [b]
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp - 1) ** 2)

    return gp


def d_loss_func(generator, discriminator, batch_z, batch_r, training):
    # detect img from generator as False
    # detect img from real img set as True
    g_img = generator(batch_z, training)
    # pass the image that generator ouputs to the discriminator
    d_gimg_logits = discriminator(g_img, training)
    # pass real images to discriminator
    d_real_logits = discriminator(batch_r, training)

    d_loss_real = cross_entropy_loss_ones(d_real_logits)
    d_loss_fake = cross_entropy_loss_zeros(d_gimg_logits)

    # gradient penalty
    gp = gradient_penalty(discriminator, batch_r, g_img)

    loss = d_loss_real + d_loss_fake + 10. * gp

    return loss, gp


def g_loss_func(generator, discriminator, batch_z, training):
    g_img = generator(batch_z, training)
    d_gimg_logits = discriminator(g_img, training)
    loss = cross_entropy_loss_ones(d_gimg_logits)

    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    z_dim = 100
    epochs = 3000000
    batch_size = 512  # same as the batch_size of real images
    d_learning_rate = 0.005
    g_learning_rate = 0.002
    training = True

    img_path = glob.glob(r'.\faces\*.jpg')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    print(dataset, img_shape)
    sample_picture = next(iter(dataset))
    print(sample_picture.shape, tf.reduce_max(sample_picture).numpy(), tf.reduce_min(sample_picture).numpy())
    dataset = dataset.repeat()
    ds_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))

    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.optimizers.RMSprop(learning_rate=g_learning_rate)
    d_optimizer = tf.optimizers.RMSprop(learning_rate=d_learning_rate)

    for epoch in range(epochs):
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_r = next(ds_iter)

        # discriminator training
        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_func(generator, discriminator, batch_z, batch_r, training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        if epoch % 5 == 0:
            with tf.GradientTape() as tape:
                g_loss = g_loss_func(generator, discriminator, batch_z, training)
            grads = tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print('Current epoch:', epoch,
                  'd_loss:', float(d_loss), 'g_loss:', float(g_loss),
                  'gp:', float(gp))

            z = tf.random.uniform([100, z_dim])
            g_imgs = generator(z, training=False)
            save_path = os.path.join('images', 'wgan-%d.png' % epoch)
            save_result(g_imgs.numpy(), 10, save_path, color_mode='P')


if __name__ == '__main__':
    main()
