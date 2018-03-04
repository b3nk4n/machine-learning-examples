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
