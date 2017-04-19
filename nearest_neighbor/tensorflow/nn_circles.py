import sys
import argparse
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles 


def main(_):
    cut = int(FLAGS.n_samples * 0.7)

    start = time.time()

    data, features = make_circles(n_samples=FLAGS.n_samples, shuffle=True, noise=0.12, factor=0.4)
    tr_data, tr_features = data[:cut], features[:cut]
    te_data, te_features = data[cut:], features[cut:]
    test = []

    fig, ax = plt.subplots()
    ax.scatter(tr_data[:, 0], tr_data[:, 1],
               marker='o', s=100, c=tr_features, cmap=plt.cm.coolwarm)
    plt.plot()
    plt.show()

    with tf.Session() as sess:
        for i, j in zip(te_data, te_features):
            distances = tf.reduce_sum(tf.square(tf.subtract(i, tr_data)), axis=1)
            neighbor = tf.arg_min(distances, 0)

            test.append(tr_features[sess.run(neighbor)])

    fig, ax = plt.subplots()
    ax.scatter(te_data[:, 0], te_data[:, 1],
               marker='o', s=100, c=test, cmap=plt.cm.coolwarm)
    plt.plot()
    plt.show()

    end = time.time()
    print("Found in %.2f seconds" % (end-start))
    print("Cluster assignments:", test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100,
                        help='The number of samples')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
