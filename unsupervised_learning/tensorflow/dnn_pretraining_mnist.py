import argparse
import sys

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import unsupervised_learning.tensorflow.models as models


def get_mnist():
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels
    Xtest, Ytest = mnist.test.images, mnist.test.labels
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)
    return Xtest, Xtrain, Ytest, Ytrain


def main(_):
    Xtest, Xtrain, Ytest, Ytrain = get_mnist()

    input_size = Xtrain.shape[-1]
    num_classes = Ytrain.shape[-1]

    if FLAGS.pretrain_model.upper() == 'AUTOENCODER':
        pretrain_model = models.AutoEncoder
    elif FLAGS.pretrain_model.upper() == 'RBM':
        pretrain_model = models.RBM
    else:
        raise Exception("Unknown pre-training model selected!")

    model = models.DNN(input_size, [256, 128, 64], num_classes, learning_rate=FLAGS.lr,
                       unsupervised_model_fn=pretrain_model)
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        model.set_session(session)
        model.fit(Xtrain, Ytrain, Xtest, Ytest, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
                  pretrain=FLAGS.pretrain, show_fig=FLAGS.show_fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='The learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='The number of training epochs')
    parser.add_argument('--pretrain', type=bool, default=True,
                        help='Whether to use unsupervised layer-wise pre-training')
    parser.add_argument('--pretrain_model', type=str, default='RBM',
                        help='Either "RBM" or "Autoencoder" as the model used for unsupervised layer-wise pre-training')
    parser.add_argument('--show_fig', type=bool, default=True,
                        help='Whether to show the learning-curve or not')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
