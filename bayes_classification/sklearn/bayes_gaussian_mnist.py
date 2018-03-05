import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from tensorflow.examples.tutorials.mnist import input_data


class BayesClassifier(object):
    """ Simple bayes classifier used for sampling from a distribution. """
    def fit(self, x, y):
        self.num_classes = len(set(y))

        self.gaussians = []
        self.p_y = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            x_k = x[y == k]
            self.p_y[k] = len(x_k)
            mean = x_k.mean(axis=0)
            covariance = np.cov(x_k.T)  # covariance shape: (784, 784)
            gaussian = {
                'mean': mean,
                'cov': covariance
            }
            self.gaussians.append(gaussian)

        # normalize p(y)
        self.p_y /= self.p_y.sum()

    def sample_given_y(self, y):
        gaussian = self.gaussians[y]
        return multivariate_normal.rvs(gaussian['mean'], gaussian['cov'])

    def sample(self):
        # randomly pick a class, while the probability is based on the occurrences in the training data
        y = np.random.choice(self.num_classes, p=self.p_y)
        return self.sample_given_y(y)


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    X, Y = mnist.train.images, mnist.train.labels

    model = BayesClassifier()
    model.fit(X, Y)

    for k in range(model.num_classes):
        # show one sample and mean image for each class
        sample = model.sample_given_y(k).reshape(28, 28)
        mean = model.gaussians[k]['mean'].reshape(28, 28)

        plt.subplot(1, 2, 1)
        plt.imshow(sample, cmap='gray')
        plt.title("Sample")
        plt.subplot(1, 2, 2)
        plt.imshow(mean, cmap='gray')
        plt.title("Mean")
        plt.show()

    # generate a random sample
    sample = model.sample().reshape(28, 28)
    plt.imshow(sample, cmap='gray')
    plt.title("Random Sample from Random Class")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
