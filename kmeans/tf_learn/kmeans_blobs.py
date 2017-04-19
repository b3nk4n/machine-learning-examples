import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization.python.ops import clustering_ops
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def get_inputs(x):
    return tf.constant(x, dtype=tf.float32), None


def main(_):
    # generate data
    centers = np.array([(-2, 2), (-2, 1.5), (1.3, -2), (2, 1.5)])
    x, y = make_blobs(n_samples=FLAGS.n_samples, centers=centers, n_features=2,
                      cluster_std=0.8, shuffle=False, random_state=42)

    kmeans = tf.contrib.learn.KMeansClustering(
        num_clusters=FLAGS.n_classes,
        model_dir="tmp/kmeans",
        initial_clusters=clustering_ops.RANDOM_INIT,
        distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE)

    kmeans.fit(input_fn=lambda: get_inputs(x), max_steps=FLAGS.max_steps)

    print("Found cluster centers: {}".format(kmeans.clusters()))
    print("GT cluster centers: {}".format(centers))

    # plot ground truth data and centers, as well as found cluster centers
    fig, ax = plt.subplots()
    ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=250)
    ax.scatter(kmeans.clusters()[:, 0], kmeans.clusters()[:, 1], marker='x', s=250)
    ax.scatter(x[:, 0], x[:, 1], marker='o', s=100, c=y, cmap=plt.cm.coolwarm)
    plt.plot()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=100,
                        help='The number of samples')
    parser.add_argument('--n_classes', type=int, default=4,
                        help='The number of classes')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='The max number of training steps')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
