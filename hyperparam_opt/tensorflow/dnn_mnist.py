import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None
NO_IMPROVEMENT_LIMIT = 2


class HyperParams(object):
    def __init__(self, lr=None, batch_size=None, n_hidden=None, keep_prob=None):
        self.lr = 0.001 if lr is None else lr
        self.batch_size = int(100 if batch_size is None else batch_size)
        self.n_hidden = int(32 if n_hidden is None else n_hidden)
        self.keep_prob = 0.5 if keep_prob is None else keep_prob

    def to_string(self):
        return 'lr: {}, batch_size: {}, n_hidden: {}, keep_prob: {}'.format(
            self.lr, self.batch_size, self.n_hidden, self.keep_prob)


def model(hyperparams):
    x = tf.placeholder(tf.float32, [None, 28*28])
    targets_ = tf.placeholder(tf.int64, [None])
    keep_prob_ = tf.placeholder_with_default(1.0, [])

    W1 = tf.Variable(tf.truncated_normal([28*28, hyperparams.n_hidden], stddev=0.01), name='W1')
    b1 = tf.Variable(tf.zeros([hyperparams.n_hidden]), name='b1')
    h = tf.nn.relu(tf.matmul(x, W1) + b1)

    h_dp = tf.nn.dropout(h, keep_prob_)

    W2 = tf.Variable(tf.truncated_normal([hyperparams.n_hidden, 10], stddev=0.01), name='W2')
    b2 = tf.Variable(tf.zeros([10]), name='b2')
    y = tf.matmul(h_dp, W2) + b2

    # Define loss
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=targets_, logits=y)

    # Define metric
    correct_prediction = tf.equal(tf.argmax(y, 1), targets_)
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return x, y, targets_, keep_prob_, accuracy_op, loss_op


def train(mnist, hyperparams):
    tf.set_random_seed(0)

    # Create the model
    x, y, targets_, keep_prob_, accuracy_op, loss_op = model(hyperparams)

    # Define optimizer
    train_op = tf.train.AdamOptimizer(hyperparams.lr).minimize(loss_op)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # Train

        epochs = 0
        batches_per_epoch = mnist.train.num_examples // hyperparams.batch_size
        best_loss = 99999.0
        no_improvement_counter = 0
        while True:
            epochs += 1
            for _ in range(batches_per_epoch):
                batch_xs, batch_ys = mnist.train.next_batch(hyperparams.batch_size)
                sess.run(train_op, feed_dict={x: batch_xs, targets_: batch_ys, keep_prob_: hyperparams.keep_prob})

            loss, accuracy = sess.run([loss_op, accuracy_op],
                                      feed_dict={x: mnist.validation.images,
                                                 targets_: mnist.validation.labels})
            print('Epoch {} > Loss: {}, Accuracy: {}'.format(epochs, loss, accuracy))

            # simple early stopping
            changed = False
            if loss < best_loss:
                best_loss = loss
                changed = True
            no_improvement_counter = 0 if changed else no_improvement_counter + 1
            if no_improvement_counter >= NO_IMPROVEMENT_LIMIT:
                break

        # TODO do a rollback here and reload the 'best' checkpoint
        loss, accuracy = sess.run([loss_op, accuracy_op],
                                  feed_dict={x: mnist.test.images,
                                             targets_: mnist.test.labels})
        return loss, accuracy, epochs


def main(_):
    hyper = HyperParams()
    hyper.lr = FLAGS.lr
    hyper.batch_size = FLAGS.batch_size
    hyper.n_hidden = FLAGS.n_hidden
    hyper.keep_prob = FLAGS.keep_prob

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    accuracy = train(mnist, hyper)
    print(accuracy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='The learning rate')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='The batch size while training')
    parser.add_argument('--n_hidden', type=int, default=128,
                        help='The number of hidden units')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='The dropout keep probability.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
