import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import unsupervised_learning.tensorflow.models as models


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels

    # convert X to binary variable
    Xtrain = (Xtrain > 0.5).astype(np.float32)

    vae = models.VariationalAutoencoder(28*28, [200, 100])
    vae.fit(Xtrain, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)

    # plot reconstruction
    done = False
    while not done:
        i = np.random.choice(len(Xtrain))
        x = Xtrain[i]
        print(x.shape)
        im = vae.posterior_predictive_sample([x]).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title('Sampled')
        plt.show()

        ans = input('Generate another?')
        if ans and ans[0] is ('n' or 'N'):
            done = True

    # plot output from random samples in latent space
    done = False
    while not done:
        im, probs = vae.prior_predictive_sample_with_probs()
        im = im.reshape(28, 28)
        probs = probs.reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(im.reshape(28, 28), cmap='gray')
        plt.title('Prior predictive sample')
        plt.subplot(1, 2, 2)
        plt.imshow(probs, cmap='gray')
        plt.title('Prior predictive probs')
        plt.show()

        ans = input('Generate another?')
        if ans and ans[0] is ('n' or 'N'):
            done = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=25,
                        help='The number of training epochs')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
