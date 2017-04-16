import sys
import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def model(input_layer_size):
    # Create the model
    x = tf.placeholder(tf.float32, [None, input_layer_size])
    weights = tf.Variable(tf.zeros([input_layer_size, 10]))
    biases = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, weights) + biases

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    return x, y, y_, cross_entropy


def main(_):
    # Import data
    mnist = input_data.read_data_sets("tmp/mnist", one_hot=True)
    input_size = mnist.train.images.shape[1]

    x, y, y_, loss = model(input_size)
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Train
        for _ in range(FLAGS.train_steps):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_value = sess.run(accuracy,
                                  feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print("Accuracy: {:.3f}".format(accuracy_value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='The intial learning rate')
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='The number of training steps')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
