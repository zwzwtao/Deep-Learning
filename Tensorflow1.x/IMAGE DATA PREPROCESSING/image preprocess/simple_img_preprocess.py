import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# read raw data of a picture
image_raw_data = tf.gfile.FastGFile("naru.jpg", 'rb').read()

with tf.Session() as sess:
    # decode the picture to get relevant 3D matrix
    img_data = tf.image.decode_jpeg(image_raw_data)

    print(img_data.get_shape())
    # print(img_data.eval())
    img_data.set_shape([1797, 2673, 3]) 
    print(img_data.get_shape())

    # show the raw picture
    plt.imshow(img_data.eval())
    plt.show()
