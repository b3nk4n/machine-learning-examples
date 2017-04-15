import time
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles 

N = 210
K = 2
# Maximum number of iterations, if the conditions are not met 
MAX_ITERATIONS = 1000
cut = int(N*0.7)

start = time.time()

data, features = make_circles(n_samples=N, shuffle=True, noise=0.12, factor=0.4)
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
