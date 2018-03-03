import argparse
import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from unsupervised_learning.tensorflow.autoencoder_mnist import AutoEncoder


class DNN(object):
    def __init__(self, num_input, hidden_layer_sizes, num_classes,
                 learning_rate, unsupervised_model_fn=AutoEncoder):
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

    def fit(self, X, Y, Xtest, Ytest, epochs, batch_size, pretrain=False):
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
        plt.plot(costs)
        plt.show()

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.encode(current_input)
            current_input = Z

        # logistic layer
        logits = tf.matmul(current_input, self.W) + self.b
        return logits


def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels
    Xtest, Ytest = mnist.test.images, mnist.test.labels
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    input_size = Xtrain.shape[-1]
    num_classes = Ytrain.shape[-1]
    model = DNN(input_size, [256, 128, 64], num_classes, learning_rate=FLAGS.lr)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.set_session(session)
        model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                  pretrain=FLAGS.pretrain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='The learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='The number of training epochs')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Whether to use unsupervised layer-wise pre-training')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
