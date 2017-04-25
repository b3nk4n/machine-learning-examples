import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def main(_):
    df = pd.read_csv('../../data/chd/chd.csv', header=0)
    print(df.describe())
    print(df['age'].mean())
    print(df['age'].std())

    x_ph = tf.placeholder(tf.float32, [None, 1], name='X')
    y_ph = tf.placeholder(tf.float32, [None, 2], name='Y')

    with tf.name_scope('Model'):
        w = tf.get_variable('W', shape=[1, 2], initializer=tf.zeros_initializer())
        b = tf.get_variable('b', shape=[2], initializer=tf.truncated_normal_initializer())

        y_model = tf.multiply(x_ph, w) + b  # we apply the tf.sigmoid later due to TensorFlow's loss function

    with tf.name_scope('CostFunction'):
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_ph, logits=y_model))

    train_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # generate a new graph
        graphnumber = 321
        plt.figure(1)

        age = df['age']

        for epoch in range(FLAGS.train_epochs):
            avg_cost = 0.0
            num_batches = len(df.values) / FLAGS.batch_size
            for i in range(num_batches):
                # transform into one-hot format
                batch_x = ((np.transpose([age]) - age.mean()) / age.std()).astype(np.float32)
                batch_y = tf.one_hot(df['chd'].values, depth=2, on_value=1, off_value=0, axis=-1)

                _, loss = sess.run([train_op, cost], feed_dict={x_ph: batch_x, y_ph: batch_y.eval()})

                avg_cost += loss / num_batches

            # display logs per epoch step
            if epoch % FLAGS.display_step == 0:
                print("Epoch: {:02d} cost: {:.5f}".format(epoch + 1, avg_cost))

                # generate a new graph and add it to the complete graph
                tr_x = np.linspace(-30, 30, 100)
                wdos = 2 * w.eval()[0][0] / age.std()
                bdos = 2 * b.eval()[0]

                # generate the prob function
                tr_y = np.exp(np.negative(wdos * tr_x) + bdos) / (np.exp(np.negative(wdos * tr_x) + bdos) + 1)

                # draw the samples and the prob function w/o normalization
                plt.subplot(graphnumber)
                graphnumber += 1

                # plot a scatter draw of the random datapoints
                plt.scatter(age, df['chd'])
                plt.plot(tr_x + age.mean(), tr_y)
                plt.grid(True)

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epochs', type=int, default=12,
                        help='The number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='The initial learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size')
    parser.add_argument('--display_step', type=int, default=2,
                        help='The step interval of intermediate values shown')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
