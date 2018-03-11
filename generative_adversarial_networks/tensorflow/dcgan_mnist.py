import argparse
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import generative_adversarial_networks.tensorflow.models as models


def get_mnist():
    mnist = input_data.read_data_sets('../../data/tmp/mnist', one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels
    Xtrain = Xtrain.astype(np.float32)
    return Xtrain, Ytrain


def main(_):
    # make dir to save samples
    if not os.path.exists('tmp'):
        os.mkdir('tmp')

    X, Y = get_mnist()
    X = X.reshape(len(X), 28, 28, 1)
    dim = X.shape[1]
    colors = X.shape[-1]

    d_sizes = {
        'conv_layers': [(2, 5, 2, False),
                        (64, 5, 2, True)],
        'dense_layers': [(1024, True)],
    }
    g_sizes = {
        'z': 100,
        'projection': 128,
        'bn_after_projection': False,
        'conv_layers': [(128, 5, 2, True),
                        (colors, 5, 2, False)],
        'dense_layers': [(1024, True)],
        'output_activation': tf.sigmoid,
    }

    model = models.DCGAN(dim, colors, d_sizes, g_sizes, FLAGS.lr, FLAGS.beta1)
    model.fit(X, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
              save_sample_interval=FLAGS.save_sample_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='The learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='The beta1 coefficient for the optimizer')
    parser.add_argument('--save_sample_interval', type=int, default=50,
                        help='The interval for saving sample images')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
