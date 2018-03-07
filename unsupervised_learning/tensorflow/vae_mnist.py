import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

import unsupervised_learning.tensorflow.models as models
import unsupervised_learning.tensorflow.utils as utils


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels

    # convert X to binary variable
    Xtrain = (Xtrain > 0.5).astype(np.float32)

    model = models.VariationalAutoencoder(28*28, [200, 100, 2])
    model.fit(Xtrain, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size)

    show_reconstruction(Xtrain, model, loop=True)
    show_sampled_from_latent_space(model, loop=True)
    visualize_latent_space(model)


def show_reconstruction(X, model, loop=False):
    done = False
    while not done:
        i = np.random.choice(len(X))
        x = X[i]
        print(x.shape)
        im = model.posterior_predictive_sample([x]).reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')
        plt.subplot(1, 2, 2)
        plt.imshow(im, cmap='gray')
        plt.title('Sampled')
        plt.show()

        if not loop:
            break
        done = utils.ask_user('Generate another?')


def show_sampled_from_latent_space(model, loop=False):
    done = False
    while not done:
        im, probs = model.prior_predictive_sample_with_probs()
        im = im.reshape(28, 28)
        probs = probs.reshape(28, 28)
        plt.subplot(1, 2, 1)
        plt.imshow(im.reshape(28, 28), cmap='gray')
        plt.title('Prior predictive sample')
        plt.subplot(1, 2, 2)
        plt.imshow(probs, cmap='gray')
        plt.title('Prior predictive probs')
        plt.show()

        if not loop:
            break
        done = utils.ask_user('Generate another?')


def visualize_latent_space(model):
    n = 20
    x_values = np.linspace(-3, 3, n)
    y_values = np.linspace(-3, 3, n)
    image = np.empty((28*n, 28*n))

    Z = []
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            z = [x, y]
            Z.append(z)
    X_recon = model.prior_predictive_with_input(Z)

    k = 0
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            x_recon = X_recon[k]
            k += 1
            x_recon = x_recon.reshape(28, 28)
            image[(n - i - 1) * 28:(n - i) * 28, j * 28:(j + 1) * 28] = x_recon
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of training epochs')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
