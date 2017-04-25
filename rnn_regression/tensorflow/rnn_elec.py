import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error

import logging
logging.basicConfig(level=logging.INFO)


def load_data(train_split=0.7, valid_split=0.2):
    df = pd.read_csv("../../data/electricity_load_diagrams/elec_load.csv", error_bad_lines=False)
    print(df.describe())
    n_data = df.values.shape[0]

    # normalize data
    train_values = df.values[:int(n_data * train_split) + 6]
    array = (df.values - train_values.mean()) / (train_values.max() - train_values.min())

    list_x = []
    list_y = []
    dataset_x = {}
    dataset_y = {}

    n_samples = 0
    for i in range(0, array.shape[0] - 6):
        list_x.append(array[i:i + 5].reshape([5, 1]))
        list_y.append(array[i + 6])
        n_samples += 1

    array_x = np.array(list_x)
    array_y = np.array(list_y)

    dataset_x['train'] = array_x[:int(n_samples * train_split)]
    dataset_x['valid'] = array_x[int(n_samples * train_split):int(n_samples * (train_split+valid_split))]
    dataset_x['test'] = array_x[int(n_samples * (train_split+valid_split)):]

    dataset_y['train'] = array_y[:int(n_samples * train_split)]
    dataset_y['valid'] = array_y[int(n_samples * train_split):int(n_samples * (train_split+valid_split))]
    dataset_y['test'] = array_y[int(n_samples * (train_split+valid_split)):]
    return dataset_x, dataset_y


def lstm_model(rnn_layers):
    def lstm_cells(layers):
        return [tf.contrib.rnn.BasicLSTMCell(layer, state_is_tuple=True)
                for layer in layers]

    def _lstm_model(x, y):
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        outputs, layers = tf.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32, time_major=False)
        output = outputs[:, -1, :]
        return learn.models.linear_regression(output, y)

    return _lstm_model


def main(_):
    rnn_layers = [5, 5]

    dataset_x, dataset_y = load_data()

    x_ph = tf.placeholder(tf.float32, [None, 5, 1], name='X')
    y_ph = tf.placeholder(tf.float32, [None, 1], name='Y')

    model = lstm_model(rnn_layers)
    y, cost = model(x_ph, y_ph)

    train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Training...")
        num_batches = int(dataset_x['train'].shape[0] / FLAGS.batch_size)
        batch_idx = 0
        epoch = 1
        for step in range(FLAGS.train_steps):
            if step % num_batches == 0 and step != 0:
                perm = np.random.permutation(dataset_x['train'].shape[0])
                dataset_x['train'] = dataset_x['train'][perm]
                dataset_y['train'] = dataset_y['train'][perm]
                batch_idx = 0
                epoch += 1

            batch_x = dataset_x['train'][batch_idx:batch_idx + FLAGS.batch_size]
            batch_y = dataset_y['train'][batch_idx:batch_idx + FLAGS.batch_size]
            batch_idx += FLAGS.batch_size

            _, loss = sess.run([train_op, cost], feed_dict={x_ph: batch_x, y_ph: batch_y})

            if step % FLAGS.display_step == 0:
                print("epoch {:02d} step {:05d} loss: {:.5f}".format(epoch, step, loss))

        print("Predictions...")
        predicted = sess.run(y, feed_dict={x_ph: dataset_x['test']})
        mse = mean_squared_error(predicted, dataset_y['test'])
        print ("MSE: {:.5f}".format(mse))

        plt.subplot()
        plot_predicted, = plt.plot(predicted, label='predicted')

        plot_test, = plt.plot(dataset_y['test'], label='test')
        plt.legend(handles=[plot_predicted, plot_test])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=10000,
                        help='The number of training steps')
    parser.add_argument('--learning_rate', type=float, default=0.025,
                        help='The initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size')
    parser.add_argument('--display_step', type=int, default=100,
                        help='The step interval of intermediate values shown')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
