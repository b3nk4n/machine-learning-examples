import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


class AutoEncoder(object):
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
            plt.plot(costs)
            plt.show()

    def transform(self, X):
        return self.session.run(self.Z, feed_dict={self.X_in: X})

    def predict(self, X):
        return self.session.run(self.X_hat, feed_dict={self.X_in: X})

    def encode(self, X):
        Z = tf.nn.sigmoid(tf.matmul(X, self.W) + self.bh)
        return Z

    def decode_logits(self, Z):
        return tf.matmul(Z, tf.transpose(self.W)) + self.bo


def show_reconstruction(x, y):
    plt.subplot(1, 2, 1)
    plt.imshow(x.reshape(28, 28), cmap='gray')
    plt.title('Original')
    plt.subplot(1, 2, 2)
    plt.imshow(y.reshape(28, 28), cmap='gray')
    plt.title('Reconstructed')
    plt.show()


def show_random_predictions(model, Xtest):
    done = False
    while not done:
        # Generate examples
        i = np.random.choice(len(Xtest))
        x = Xtest[i]
        y = model.predict([x])
        show_reconstruction(x, y)

        ans = input("Generate another?")
        if ans and ans[0] in ('n' or 'N'):
            done = True


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels
    Xtest, Ytest = mnist.test.images, mnist.test.labels
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    input_size = Xtrain.shape[-1]
    model = AutoEncoder(input_size, 256, learning_rate=FLAGS.lr, id=0)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.set_session(session)
        model.fit(Xtrain, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size, show_fig=True)
        show_random_predictions(model, Xtest)


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
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
