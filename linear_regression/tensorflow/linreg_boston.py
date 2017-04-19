import sys
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle


def main(_):
    # read data
    df = pd.read_csv('../../data/boston/boston_train.csv', header=0)
    print(df.describe())

    f, ax1 = plt.subplots()
    for i in range(1, 8):
        number = 420 + i
        ax1.locator_params(nbins=3)
        ax1 = plt.subplot(number)
        plt.title(list(df)[i])
        ax1.scatter(df[df.columns[i]], df['MEDV'])
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()

    x_ph = tf.placeholder(tf.float32, name='X')
    y_ph = tf.placeholder(tf.float32, name='Y')

    with tf.name_scope('Model'):
        w = tf.get_variable('W', shape=[2], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('b', shape=[2], initializer=tf.truncated_normal_initializer())

        y_model = tf.multiply(x_ph, w) + b

    with tf.name_scope('CostFunction'):
        cost = tf.reduce_mean(tf.pow(y_ph - y_model, 2))

    train_op = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def print_params(w_op, b_op):
            b_val = sess.run(b_op)
            w_val = sess.run(w_op)
            print('w: {} b: {}'.format(w_val, b_val))

        def plot_params(w_op, b_op, x_data, y_data):
            b_val = sess.run(b_op)
            w_val = sess.run(w_op)
            plt.scatter(x_data[:, 0], y_data, marker='o')
            plt.scatter(x_data[:, 1], y_data, marker='x')
            plt.plot(x_data, w_val * x_data + b_val)
            plt.show()

        # x=[INDUS, AGE] y=[MEDV]
        x_values = df[['INDUS', 'AGE']].values.astype(float)
        y_values = df['MEDV'].values.astype(float)
        print_params(w, b)
        plot_params(w, b, x_values, y_values)

        for a in range(1, FLAGS.train_steps + 1):
            cost_sum = 0.0
            for i, j in zip(x_values, y_values):
                _, cost_val = sess.run([train_op, cost], feed_dict={x_ph: i, y_ph: j})
                cost_sum += cost_val

            x_values, y_values = shuffle(x_values, y_values)

            if a % 5 == 0:
                print('@{:-3d}: {:.3f}'.format(a, cost_sum / x_values.shape[0]))

        print_params(w, b)
        plot_params(w, b, x_values, y_values)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='The initial learning rate')
    parser.add_argument('--train_steps', type=int, default=100,
                        help='The number of training steps')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
