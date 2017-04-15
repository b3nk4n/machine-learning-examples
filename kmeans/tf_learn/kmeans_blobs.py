import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization.python.ops import clustering_ops
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

K = 4
N = 100
MAX_ITERATIONS = 1000


# generate data
centers = np.array([(-2, 2), (-2, 1.5), (1.3, -2), (2, 1.5)])
X, y = make_blobs(n_samples=N, centers=centers, n_features=2,
                  cluster_std=0.8, shuffle=False, random_state=42)


def get_inputs(x):
    return tf.constant(x, dtype=tf.float32), None


kmeans = tf.contrib.learn.KMeansClustering(
    num_clusters=K,
    model_dir="tmp/kmeans",
    initial_clusters=clustering_ops.RANDOM_INIT,
    distance_metric=clustering_ops.SQUARED_EUCLIDEAN_DISTANCE)

kmeans.fit(input_fn=lambda: get_inputs(X), max_steps=MAX_ITERATIONS)

print("Found cluster centers: {}".format(kmeans.clusters()))
print("GT cluster centers: {}".format(centers))

# plot ground truth data and centers, as well as found cluster centers
fig, ax = plt.subplots()
ax.scatter(centers[:, 0], centers[:, 1], marker='o', s=250)
ax.scatter(kmeans.clusters()[:, 0], kmeans.clusters()[:, 1], marker='x', s=250)
ax.scatter(X[:, 0], X[:, 1], marker='o', s=100, c=y, cmap=plt.cm.coolwarm)
plt.plot()
plt.show()
