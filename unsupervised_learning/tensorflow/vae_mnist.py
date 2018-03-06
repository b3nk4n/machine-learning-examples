import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import unsupervised_learning.tensorflow.utils as utils
st = tf.contrib.bayesflow.stochastic_tensor


class DenseLayer(object):
    def __init__(self, m, n, activation=lambda x: x):
        self.W = tf.Variable(tf.random_normal((m, n)) * 2 / np.sqrt(m))
        self.b = tf.Variable(tf.zeros(n), dtype=tf.float32)
        self.activation = activation

    def forward(self, X):
        return self.activation(tf.matmul(X, self.W) + self.b)


class VariationalAutoencoder(object):
    SMOOTHING_EPSILON = 1e-6  # to not get a number too close to zero, which would cause a singularity

    def __init__(self, num_input, num_hiddens):
        self.X = tf.placeholder(tf.float32, shape=[None, num_input])

        #encoder
        self.encoder_layers = []
        current_input_size = num_input
        for current_output_size in num_hiddens[:1]:
            layer = DenseLayer(current_input_size, current_input_size,
                               activation=tf.nn.relu)
            self.encoder_layers.append(layer)
            current_input_size = current_output_size

        num_z = num_hiddens[-1]

        final_encoder_layer = DenseLayer(current_input_size, 2 * num_z)  # 2 * num_z, because z_i = (mean, std-dev)
        self.encoder_layers.append(final_encoder_layer)

        current_layer_value = self.X
        for layer in self.encoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        self.means = current_layer_value[:, :num_z]
        # use softplus to ensure std-dev is not negative
        self.stddev = tf.nn.softplus(current_layer_value[:, num_z:]) + self.SMOOTHING_EPSILON

        with st.value_type(st.SampleValue()):
            # this returns q(Z), the distribution of the latent variable Z
            self.Z = st.StochasticTensor(tf.distributions.Normal(loc=self.means, scale=self.stddev))

        # decoder
        self.decoder_layers = []
        current_input_size = num_z
        for current_output_size in reversed(num_hiddens[:-1]):
            layer = DenseLayer(current_input_size, current_output_size,
                               activation=tf.nn.relu)
            self.decoder_layers.append(layer)
            current_input_size = current_output_size

        final_decoder_layer = DenseLayer(current_input_size, num_input)
        self.decoder_layers.append(final_decoder_layer)

        # logits
        current_layer_value = self.Z
        for layer in self.decoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        logits = current_layer_value
        posterior_predictive_logits = logits

        self.X_hat_distribution = tf.distributions.Bernoulli(logits=logits)

        # take a sample from X_hat, which is called the posterior predictive sample
        self.posterior_predictive = self.X_hat_distribution.sample()
        self.posterior_predictive_probs = tf.nn.sigmoid(logits)

        # take a sample from a Z ~ N(0, 1) and feed it through the decoder, called the prior predictive sample
        standard_normal = tf.distributions.Normal(
            loc=np.zeros(num_z, dtype=np.float32),
            scale=np.ones(num_z, dtype=np.float32)
        )

        z_std = standard_normal.sample(1)
        current_layer_value = z_std
        for layer in self.decoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        logits = current_layer_value

        prior_predictive_dist = tf.distributions.Bernoulli(logits=logits)
        self.prior_predictive = prior_predictive_dist.sample()
        self.prior_predictive_probs = tf.nn.sigmoid(logits)

        # prior preditive from given input used for generating visualization
        self.Z_input = tf.placeholder(tf.float32, shape=[None, num_z])
        current_layer_value = self.Z_input
        for layer in self.decoder_layers:
            current_layer_value = layer.forward(current_layer_value)
        logits = current_layer_value
        self.prior_predictive_from_input_probs = tf.nn.sigmoid(logits)

        # cost function
        kl = tf.reduce_sum(tf.distributions.kl_divergence(self.Z.distribution, standard_normal), axis=1)
        expected_log_likelihood = tf.reduce_sum(self.X_hat_distribution.log_prob(self.X), axis=1)
        # equals:
        # expected_log_likelihood = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=self.X,
        #     logits=posterior_predictive_logits
        # ), axis=1)
        self.elbo = tf.reduce_sum(expected_log_likelihood - kl)

        self.train_op = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(self.elbo)

        # setup session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, epochs, batch_size):
        costs = []
        n_batches = len(X) // batch_size
        print('n_batches: {}'.format(n_batches))
        for epoch in range(epochs):
            print('epoch: {}'.format(epoch))
            np.random.shuffle(X)
            for b in range(n_batches):
                batch = X[b*batch_size:(b+1)*batch_size]
                _, cost = self.sess.run([self.train_op, self.elbo], feed_dict={
                    self.X: batch
                })
                cost /= batch_size  # FIXME use tf.reduce_mean?
                costs.append(cost)
                if b % 100 == 0:
                    print('@{:4d} > cost: {:.3f}'.format(b, cost))
        utils.show_costs(costs)

    def transform(self, X):
        return self.sess.run(self.means, feed_dict={
            self.X: X
        })

    def prior_predictive_with_input(self, Z):
        return self.sess.run(
            self.prior_predictive_from_input_probs, feed_dict={
                self.Z_input: Z
            }
        )

    def posterior_predictive_sample(self, X):
        """Returns a sample from p(x_new | X)."""
        return self.sess.run(self.posterior_predictive, feed_dict={
            self.X: X
        })

    def prior_predictive_sample_with_probs(self):
        """Returns a sample from p(x_new | z), where z ~ N(0, 1)."""
        return self.sess.run([self.prior_predictive, self.prior_predictive_probs])


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels

    # convert X to binary variable
    Xtrain = (Xtrain > 0.5).astype(np.float32)

    vae = VariationalAutoencoder(28*28, [200, 100])
    vae.fit(Xtrain)

    # plot reconstruction
    done = False
    while not done:
        i = np.random.choice(len(Xtrain))
        x = Xtrain[i]
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
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
