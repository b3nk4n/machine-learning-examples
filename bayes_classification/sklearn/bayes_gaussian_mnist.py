import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture
from tensorflow.examples.tutorials.mnist import input_data


class MultivariateNormalDistribution(object):
    def fit(self, x_k):
        mean = x_k.mean(axis=0)
        covariance = np.cov(x_k.T)  # covariance shape: (784, 784)
        gaussian = {
            'mean': mean,
            'cov': covariance
        }
        return gaussian

    def sample_with_mean(self, gaussian):
        sample = multivariate_normal.rvs(gaussian['mean'], gaussian['cov'])
        mean = gaussian['mean']  # just for debug
        return sample, mean


class GaussianMixtureDistribution(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, x_k):
        gmm = BayesianGaussianMixture(self.n_components)
        gmm.fit(x_k)
        return gmm

    def sample_with_mean(self, gmm):
        sample, cluster_k = gmm.sample()
        mean = gmm.means_[cluster_k]  # just for debug
        return sample, mean


class BayesClassifier(object):
    """ Simple bayes classifier used for sampling from a distribution. """
    def __init__(self, sampling_distribution):
        self.sampling_distribution = sampling_distribution

    def fit(self, x, y):
        self.num_classes = len(set(y))

        self.gaussians = []
        self.p_y = np.zeros(self.num_classes)
        for k in range(self.num_classes):
            x_k = x[y == k]
            self.p_y[k] = len(x_k)
            distribution = self.sampling_distribution.fit(x_k)
            self.gaussians.append(distribution)

        # normalize p(y)
        self.p_y /= self.p_y.sum()

    def sample_given_y(self, y):
        gaussian = self.gaussians[y]
        sample, mean = self.sampling_distribution.sample_with_mean(gaussian)
        return sample, mean

    def sample(self):
        # randomly pick a class, while the probability is based on the occurrences in the training data
        y = np.random.choice(self.num_classes, p=self.p_y)
        sample, _ = self.sample_given_y(y)
        return sample


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    X, Y = mnist.train.images, mnist.train.labels

    if FLAGS.distribution.upper() == 'MVN':
        sampling_distribution = MultivariateNormalDistribution()
    elif FLAGS.distribution.upper() == 'GMM':
        sampling_distribution = GaussianMixtureDistribution(n_components=10)
    else:
        raise Exception('Unknown sampling distribution was selected!')

    model = BayesClassifier(sampling_distribution)
    model.fit(X, Y)

    for k in range(model.num_classes):
        # show one sample and mean image for each class
        sample, mean = model.sample_given_y(k)
        sample = sample.reshape(28, 28)
        mean = mean.reshape(28, 28)

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
    parser.add_argument('--distribution', type=str, default='MVN',
                        help='The distribution to sample from: GMM (gaussian mixture) or MVN (multi-variate normal)')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
