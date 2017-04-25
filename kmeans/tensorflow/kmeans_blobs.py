import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs


def main(_):
    k = FLAGS.n_classes
    n = FLAGS.n_samples
    start = time.time()

    centers = np.array([(-2, 2), (-2, 1.5), (1.3, -2), (2, 1.5)])
    x, y = make_blobs(n_samples=n, centers=centers, n_features=2,
                      cluster_std=0.8, shuffle=False, random_state=42)

    fig, ax = plt.subplots()
    ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=250)
    ax.scatter(x[:, 0], x[:, 1], marker='o', s=100, c=y, cmap=plt.cm.coolwarm)
    plt.plot()
    plt.show()

    points = tf.Variable(x)
    cluster_assignments = tf.Variable(tf.zeros([n], dtype=tf.int64))
    # use first K points as starting-centroids
    centroids = tf.Variable(points.initialized_value()[0:k, :])  # centroid == cluster center

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # calculate the distances between each point and every centroid, for each dimension
    rep_centroids = tf.reshape(tf.tile(centroids, [n, 1]), [n, k, 2])
    rep_points = tf.reshape(tf.tile(points, [1, k]), [n, k, 2])
    sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), axis=2)
    best_centroids = tf.argmin(sum_squares, 1)  # shape: [N], the index of the closest centroid

    did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))

    def bucket_mean(data, bucket_ids, num_buckets):
        total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
        count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
        return total / count

    # K means
    means = bucket_mean(points, best_centroids, k)

    with tf.control_dependencies([did_assignments_change]):
        do_updates = tf.group(
            centroids.assign(means),
            cluster_assignments.assign(best_centroids))

    iteration = 0
    changed = True
    assignments = None
    while changed and iteration < FLAGS.max_steps:
        fig, ax = plt.subplots()
        iteration += 1
        [changed, _] = sess.run([did_assignments_change, do_updates])

        # plot current centers + assignments
        [centers, assignments] = sess.run([centroids, cluster_assignments])
        ax.scatter(sess.run(points)[:, 0], sess.run(points)[:, 1],
                   marker='o', s=200, c=assignments, cmap=plt.cm.coolwarm)
        ax.scatter(centers[:, 0], centers[:, 1], marker='^', s=550, c=[2, 1, 4, 3], cmap=plt.cm.plasma)
        ax.set_title('iterations ' + str(iteration))

        if not os.path.isdir("out"):
            os.mkdir("out")
        plt.savefig("out/kmeans" + str(iteration) + ".png")

    print("Finished after {} iterations.".format(iteration))

    end = time.time()
    print("Found in {:.2f} seconds (iteration: {})".format(end - start, iteration))
    print("Centroids:")
    print(centers)
    print("Cluster assignments:", assignments)


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
