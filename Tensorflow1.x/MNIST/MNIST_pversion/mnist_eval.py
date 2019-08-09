import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# load the latest model every 10 seconds then evaluate it
EVAL_INTERVAL_SECS = 10

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels,
        }

        # don't need regularization
        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        # print(variable_to_restore)
        saver = tf.train.Saver(variables_to_restore)

        counter = 1
        while True:
            counter = counter + 1
            if(counter >= 3):
                return
            with tf.Session() as sess:
                # find the latest file in savepath through checkpoint file
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, "MNIST_model/mnist_model-29001")
                    # e.g. mnist_model-29001.data-00000-of-00001
                    print(ckpt.model_checkpoint_path)
                    global_step = 29001
                    # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training steps, validation accuracy is %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return

            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST_model/", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()


























