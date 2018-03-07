import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle

import unsupervised_learning.tensorflow.utils as utils


class AutoEncoder(object):
    """ Simple autoencoder model used for unsupervised pre-training. """
    def __init__(self, num_input, num_hidden, learning_rate, id):
        self.num_hidden = num_hidden
        self.id = id
        self.build(num_input, num_hidden, learning_rate)

    def set_session(self, session):
        self.session = session

    def build(self, num_input, num_hidden, learning_rate):
        self.W = tf.Variable(tf.random_normal(shape=(num_input, num_hidden)))
        self.bh = tf.Variable(np.zeros(num_hidden).astype(np.float32))
        self.bo = tf.Variable(np.zeros(num_input).astype(np.float32))

        self.X_in = tf.placeholder(tf.float32, shape=(None, num_input), name='X_in')
        self.Z = self.encode(self.X_in)
        logits = self.decode_logits(self.Z)
        self.X_hat = tf.nn.sigmoid(logits)

        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X_in,
                logits=logits))

        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def fit(self, X, epochs, batch_size, show_fig=False):
        num_examples = X.shape[0]
        n_batches = num_examples // batch_size

        costs = []
        print("training autoencoder: %s" % self.id)
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j*batch_size:(j*batch_size + batch_size)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
                costs.append(c)
        if show_fig:
            utils.show_costs(costs)

    def transform(self, X):
        return self.session.run(self.Z, feed_dict={self.X_in: X})

    def predict(self, X):
        return self.session.run(self.X_hat, feed_dict={self.X_in: X})

    def encode(self, X):
        Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.bh)
        return Z

    def decode_logits(self, Z):
        return tf.matmul(Z, tf.transpose(self.W)) + self.bo


class RBM(object):
    """ Restricted Boltzman Machine for unsupervised pre-training. """
    def __init__(self, num_input, num_hidden, learning_rate, id):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.id = id
        self.build(num_input, num_hidden, learning_rate)

    def set_session(self, session):
        self.session = session

    def build(self, num_input, num_hidden, learning_rate):
        # params
        self.W = tf.Variable(tf.random_normal(shape=(num_input, num_hidden)) * np.sqrt(2.0 / num_hidden))
        # note: without limiting variance, you get numerical stability issues
        self.c = tf.Variable(np.zeros(num_hidden).astype(np.float32))
        self.b = tf.Variable(np.zeros(num_input).astype(np.float32))

        # data
        self.X_in = tf.placeholder(tf.float32, shape=(None, num_input))

        # conditional probabilities (also possible to do this using tf.contrib.distributions.Bernoulli)
        visible_layer = self.X_in
        self.p_h_given_v = tf.nn.sigmoid(tf.matmul(visible_layer, self.W) + self.c)
        r = tf.random_uniform(shape=tf.shape(self.p_h_given_v))
        hidden_layer = tf.to_float(r < self.p_h_given_v)

        p_v_given_h = tf.nn.sigmoid(tf.matmul(hidden_layer, tf.transpose(self.W)) + self.b)
        r = tf.random_uniform(shape=tf.shape(p_v_given_h))
        X_sample = tf.to_float(r < p_v_given_h)

        # build the objective
        objective = tf.reduce_mean(self.free_energy(self.X_in)) - tf.reduce_mean(self.free_energy(X_sample))
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(objective)

        # build the cost (not used for optimization, just for output and verification during training)
        Z = self.encode(self.X_in)
        logits = self.decode_logits(Z)
        self.cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.X_in,
                logits=logits))

    def fit(self, X, epochs, batch_size, show_fig=False):
        num_examples, input_size = X.shape
        n_batches = num_examples // batch_size

        costs = []
        print("training rbm: %s" % self.id)
        for i in range(epochs):
            print("epoch:", i)
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_size:(j * batch_size + batch_size)]
                _, c = self.session.run((self.train_op, self.cost), feed_dict={self.X_in: batch})
                if j % 10 == 0:
                    print("j / n_batches:", j, "/", n_batches, "cost:", c)
                costs.append(c)
        if show_fig:
            utils.show_costs(costs)

    def free_energy(self, V):
        b = tf.reshape(self.b, (self.num_input, 1))
        first_term = -tf.matmul(V, b)
        first_term = tf.reshape(first_term, (-1,))

        second_term = -tf.reduce_sum(
            # tf.log(1 + tf.exp(tf.matmul(V, self.W) + self.c)),
            tf.nn.softplus(tf.matmul(V, self.W) + self.c),
            axis=1)

        return first_term + second_term

    def encode(self, X):
        Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.c)
        return Z

    def decode_logits(self, Z):
        return tf.matmul(Z, tf.transpose(self.W)) + self.b

    def transform(self, X):
        return self.session.run(self.p_h_given_v, feed_dict={self.X_in: X})


class DNN(object):
    """ Simple multi-layer neural network, which uses an unsupervised model for pre-training. """
    def __init__(self, num_input, hidden_layer_sizes, num_classes,
                 learning_rate, unsupervised_model_fn):
        self.hidden_layers = []
        input_size = num_input
        for i, output_size in enumerate(hidden_layer_sizes):
            ae = unsupervised_model_fn(input_size, output_size, learning_rate, i)
            self.hidden_layers.append(ae)
            input_size = output_size
        self.build_final_layer(num_input, hidden_layer_sizes[-1], num_classes, learning_rate)

    def set_session(self, session):
        self.session = session
        for layer in self.hidden_layers:
            layer.set_session(session)

    def build_final_layer(self, num_input, num_hidden, num_classes, learning_rate):
        # initialize logistic regression layer
        self.W = tf.Variable(tf.random_normal(shape=(num_hidden, num_classes)))
        self.b = tf.Variable(np.zeros(num_classes).astype(np.float32))

        self.X = tf.placeholder(tf.float32, shape=(None, num_input), name='X')
        labels = tf.placeholder(tf.int32, shape=(None,), name='labels')
        self.Y = labels
        logits = self.forward(self.X)

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.prediction = tf.argmax(logits, 1)

    def fit(self, X, Y, Xtest, Ytest, epochs, batch_size, pretrain=False, show_fig=False):
        num_examples = len(X)

        print("greedy layer-wise training of autoencoders...")
        pretrain_epochs = 1
        if not pretrain:
            pretrain_epochs = 0

        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs, batch_size=batch_size)

            # create current_input for the next layer
            current_input = ae.transform(current_input)

        n_batches = num_examples // batch_size
        costs = []
        print("supervised training...")
        for i in range(epochs):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j * batch_size:(j * batch_size + batch_size)]
                Ybatch = Y[j * batch_size:(j * batch_size + batch_size)]
                self.session.run(
                    self.train_op,
                    feed_dict={self.X: Xbatch, self.Y: Ybatch}
                )
                if j % 10 == 0:
                    c, p = self.session.run(
                        (self.cost, self.prediction),
                        feed_dict={self.X: Xtest, self.Y: Ytest})
                    error_rate = np.mean(p != Ytest)
                    print("j / n_batches:", j, "/", n_batches, "cost:", c, "error:", error_rate)
                    costs.append(c)
        if show_fig:
            utils.show_costs(costs)

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.encode(current_input)
            current_input = Z

        # logistic layer
        logits = tf.matmul(current_input, self.W) + self.b
        return logits


class DenseLayer(object):
    def __init__(self, m, n, activation=lambda x: x):
        self.W = tf.Variable(tf.random_normal((m, n)) * 2 / np.sqrt(m))
        self.b = tf.Variable(tf.zeros(n), dtype=tf.float32)
        self.activation = activation

    def forward(self, X):
        return self.activation(tf.matmul(X, self.W) + self.b)


class VariationalAutoencoder(object):
    """
    Simple variational autoencoder implementation.

    This implementations is based on:
    https://deeplearningcourses.com/c/deep-learning-gans-and-variational-autoencoders
    """
    SMOOTHING_EPSILON = 1e-6  # to not get a number too close to zero, which would cause a singularity

    def __init__(self, num_input, num_hiddens):
        self.X = tf.placeholder(tf.float32, shape=[None, num_input])

        #encoder
        self.encoder_layers = []
        current_input_size = num_input
        for current_output_size in num_hiddens[:1]:
            layer = DenseLayer(current_input_size, current_output_size,
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

        # @deprecated since TF r1.5
        # with st.value_type(st.SampleValue()):
        #     # this returns q(Z), the distribution of the latent variable Z
        #     self.Z = st.StochasticTensor(tf.distributions.Normal(loc=self.means, scale=self.stddev))

        self.Z = tf.distributions.Normal(loc=self.means, scale=self.stddev).sample()

        # alternative A: to the previous, but using the "reparameterization trick"
        #standard_normal = tf.distributions.Normal(
        #    loc=np.zeros(num_z, dtype=np.float32),
        #    scale=np.ones(num_z, dtype=np.float32)
        #)
        #e = standard_normal.sample(tf.shape(self.means)[0])
        #self.Z = e * self.stddev + self.means

        # alternative B:
        #eps = tf.random_normal((tf.shape(self.X)[0], num_z), 0, 1,
        #                       dtype=tf.float32)
        # z = sigma*epsilon + mu
        # self.Z = tf.sqrt(tf.exp(self.stddev)) * eps + self.means


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
        kl = -tf.log(self.stddev) + 0.5 * (self.stddev ** 2 + self.means ** 2) - 0.5
        kl = tf.reduce_sum(kl, axis=1)
        # equals (before TF r1.5):
        # kl = tf.reduce_sum(tf.distributions.kl_divergence(self.Z, standard_normal), axis=1)

        expected_log_likelihood = tf.reduce_sum(-tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.X,
            logits=posterior_predictive_logits
        ), axis=1)
        # equals:
        # expected_log_likelihood = tf.reduce_sum(self.X_hat_distribution.log_prob(self.X), axis=1)

        elbo = tf.reduce_mean(expected_log_likelihood - kl)
        self.cost = -elbo

        self.train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)

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
                _, cost = self.sess.run([self.train_op, self.cost], feed_dict={
                    self.X: batch
                })
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
