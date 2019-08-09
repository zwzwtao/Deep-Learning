import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# read raw data of a picture
image_raw_data = tf.gfile.FastGFile("naru.jpg", 'rb').read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    # add bounding box
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    # sample_ditorted_bounding_box requires real number input picture
    image_float = tf.image.convert_image_dtype(img_data, tf.float32)

    batched_img = tf.expand_dims(image_float, 0)
    image_with_box = tf.image.draw_bounding_boxes(batched_img, boxes)
    plt.imshow(image_with_box[0].eval())
    plt.show()

    # randomly crop the picture, min_object_covered=0.4 means the crop part should cover
    # at least 40% of the bounding box
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image_float), bounding_boxes=boxes, min_object_covered=0.4
    )

    # distorted image
    distorted_image = tf.slice(image_float, begin, size)
    plt.imshow(distorted_image.eval())
    plt.show()

    image_small = tf.image.resize_images(image_float, [180, 267], method=0)
    batched_img = tf.expand_dims(image_small, 0)
    image_with_box = tf.image.draw_bounding_boxes(batched_img, bbox_for_draw)
    print(bbox_for_draw.eval())
    plt.imshow(image_with_box[0].eval())
    plt.show()
