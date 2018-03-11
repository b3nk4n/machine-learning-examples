import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import unsupervised_learning.tensorflow.models as models
import unsupervised_learning.tensorflow.utils as utils


def get_mnist():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels
    Xtest, Ytest = mnist.test.images, mnist.test.labels
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    return Xtest, Xtrain


def show_random_predictions(model, Xtest, loop=False):
    done = False
    while not done:
        # Generate examples
        i = np.random.choice(len(Xtest))
        x = Xtest[i]
        y = model.predict([x])
        utils.show_reconstruction(x, y)

        if not loop:
            break
        done = utils.ask_user('Generate another?')


def main(_):
    Xtest, Xtrain = get_mnist()

    input_size = Xtrain.shape[-1]
    model = models.AutoEncoder(input_size, 256, learning_rate=FLAGS.lr, id=0)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.set_session(session)
        model.fit(Xtrain, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, show_fig=FLAGS.show_fig)
        show_random_predictions(model, Xtest, loop=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--lr', type=float, default=1e-1,
                        help='The learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The number of training epochs')
    parser.add_argument('--show_fig', type=bool, default=False,
                        help='Whether to show the learning-curve or not')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
