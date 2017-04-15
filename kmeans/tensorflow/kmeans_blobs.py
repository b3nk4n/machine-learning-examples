import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

K = 4
N = 100
MAX_ITERATIONS = 1000

start = time.time()

centers = np.array([(-2, 2), (-2, 1.5), (1.3, -2), (2, 1.5)])
X, y = make_blobs(n_samples=N, centers=centers, n_features=2,
                  cluster_std=0.8, shuffle=False, random_state=42)

fig, ax = plt.subplots()
ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=250)
plt.show()

fig, ax = plt.subplots()
ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=250)
ax.scatter(X[:, 0], X[:, 1], marker='o', s=100, c=y, cmap=plt.cm.coolwarm)
plt.plot()

points = tf.Variable(X)
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))
# use first K points as starting-centroids
centroids = tf.Variable(points.initialized_value()[0:K, :])  # centroid == cluster center

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# calculate the distances between each point and every centroid, for each dimension
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), axis=2)
best_centroids = tf.argmin(sum_squares, 1)  # shape: [N], the index of the closest centroid

did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))


def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    print("bucket-mean total", sess.run(total))
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    print("bucket-mean count: ", sess.run(count))
    return total / count

# K means
means = bucket_mean(points, best_centroids, K)

print("means shape: ", means.get_shape().as_list())

with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(
        centroids.assign(means),
        cluster_assignments.assign(best_centroids))


fig, ax = plt.subplots()

iteration = 0
changed = True
while changed and iteration < MAX_ITERATIONS:
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

points_val = sess.run(points)
ax.scatter(points_val[:, 0], points_val[:, 1],
           marker='o', s=200, c=assignments, cmap=plt.cm.coolwarm)

end = time.time()
print("Found in %.2f seconds" % (end - start)), iteration, "iterations"
print("Centroids:")
print(centers)
print("Cluster assignments:", assignments)

