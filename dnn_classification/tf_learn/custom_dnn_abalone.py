from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import tempfile
from six.moves import urllib

import numpy as np
import tensorflow as tf
import sklearn.metrics

tf.logging.set_verbosity(tf.logging.INFO)


def maybe_download(train_data, test_data, predict_data):
    """Maybe downloads training data and returns train and test file names."""
    def _download(url):
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve(url, tmp_file.name)
        file_name = tmp_file.name
        tmp_file.close()
        return file_name

    if train_data:
        train_file_name = train_data
    else:
        train_file_name = _download("http://download.tensorflow.org/data/abalone_train.csv")
        print("Training data is downloaded to %s" % train_file_name)

    if test_data:
        test_file_name = test_data
    else:
        test_file_name = _download("http://download.tensorflow.org/data/abalone_test.csv")
        print("Test data is downloaded to %s" % test_file_name)

    if predict_data:
        predict_file_name = predict_data
    else:
        predict_file_name = _download("http://download.tensorflow.org/data/abalone_predict.csv")
        print("Prediction data is downloaded to %s" % predict_file_name)

    return train_file_name, test_file_name, predict_file_name


class CustomModel:
    LEARNING_RATE_KEY = "learning_rate"
    DROPOUT_KEY = "dropout"

    def __init__(self, learning_rate, dropout):
        self.learning_rate = learning_rate
        self.dropout = dropout

    def params(self):
        return {
            self.LEARNING_RATE_KEY: self.learning_rate,
            self.DROPOUT_KEY: self.dropout
        }

    def build(self):
        def _build(features, labels, mode, params):
            # 1. Configure the model via TensorFlow operations
            # input_layer = tf.contrib.layers.input_from_feature_columns(
            #     columns_to_tensors=features, feature_columns=[age, height, weight])

            y = tf.contrib.layers.fully_connected(features, num_outputs=10, activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())
            y = tf.contrib.layers.fully_connected(y, num_outputs=10, activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())
            y = tf.contrib.layers.fully_connected(y, num_outputs=10, activation_fn=tf.nn.relu,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())
            keep_prob = params[self.DROPOUT_KEY] if mode == tf.estimator.ModeKeys.TRAIN else 1.0
            y = tf.nn.dropout(y, keep_prob=keep_prob)
            y = tf.contrib.layers.fully_connected(y, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                                  weights_initializer=tf.contrib.layers.xavier_initializer())

            # Reshape output layer to 1-dim Tensor to return predictions
            predictions = tf.reshape(y, [-1])

            # 2. Define the loss function for training/evaluation
            loss = None
            eval_metric_ops = None

            if mode != tf.estimator.ModeKeys.PREDICT:
                loss = tf.losses.mean_squared_error(labels, predictions)

                reshaped_labels = tf.reshape(labels, [-1])
                reshaped_labels = tf.cast(reshaped_labels, tf.float32)
                reshaped_preds = tf.reshape(predictions, [-1])
                reshaped_preds = tf.round(reshaped_preds)
                eval_metric_ops = {
                    "rmse": tf.metrics.root_mean_squared_error(
                        tf.cast(labels, tf.float32), predictions),
                    "accuracy": tf.metrics.accuracy(reshaped_labels, reshaped_preds),
                    "precision": tf.metrics.precision(reshaped_labels, reshaped_preds)
                }

            # 3. Define the training operation/optimizer
            train_op = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.contrib.layers.optimize_loss(
                    loss=loss,
                    global_step=tf.contrib.framework.get_global_step(),
                    learning_rate=params[self.LEARNING_RATE_KEY],
                    optimizer='SGD')

            # 4. Generate predictions
            predictions_dict = None
            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions_dict = {'ages': predictions}

            # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
            return tf.estimator.EstimatorSpec(mode, predictions_dict, loss, train_op, eval_metric_ops)
        return _build


def batched_input_fn(dataset_x, dataset_y, batch_size, num_epochs=None, shuffle=True):
    def _input_fn():
        all_x = tf.constant(dataset_x, shape=dataset_x.shape, dtype=tf.float32)
        datasets = [all_x]
        if dataset_y is not None:
            all_y = tf.constant(dataset_y, shape=dataset_y.shape, dtype=tf.float32)
            datasets.append(all_y)
        sliced_input = tf.train.slice_input_producer(datasets, num_epochs=num_epochs, shuffle=shuffle)
        return tf.train.batch(sliced_input, batch_size=batch_size, num_threads=4)
    return _input_fn


def main(_):
    # Load datasets
    abalone_train, abalone_test, abalone_predict = maybe_download(
        FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

    # Training examples
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_train, target_dtype=np.int, features_dtype=np.float32)

    # Test examples
    test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_test, target_dtype=np.int, features_dtype=np.float32)

    # Set of 7 examples for which to predict abalone_snail ages
    prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=abalone_predict, target_dtype=np.int, features_dtype=np.float32)

    m = CustomModel(FLAGS.learning_rate, FLAGS.keep_prob)
    nn = tf.estimator.Estimator(model_fn=m.build(), params=m.params())

    nn.train(input_fn=batched_input_fn(training_set.data, training_set.target,
                                       batch_size=FLAGS.batch_size),
             steps=FLAGS.train_steps)

    ev = nn.evaluate(input_fn=batched_input_fn(test_set.data, test_set.target,
                                               batch_size=FLAGS.batch_size, num_epochs=1))
    print("Loss: {}".format(ev["loss"]))
    print("RMSE: {}".format(ev["rmse"]))

    # Print out predictions
    pred_generator = nn.predict(input_fn=batched_input_fn(prediction_set.data, None,
                                                          batch_size=FLAGS.batch_size, num_epochs=1, shuffle=False))
    pred_list = [p['ages'] for p in pred_generator]
    pred_array = np.asarray(pred_list)
    pred_array = np.round(pred_array)

    cm = sklearn.metrics.confusion_matrix(y_true=prediction_set.target[:pred_array.shape[0]],
                                          y_pred=pred_array)
    print(cm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--train_data", type=str, default="",
                        help="Path to the training data.")
    parser.add_argument("--test_data", type=str, default="",
                        help="Path to the test data.")
    parser.add_argument("--predict_data", type=str, default="",
                        help="Path to the prediction data.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="The initial learning rate.")
    parser.add_argument("--train_steps", type=int, default=1000,
                        help="The number of training steps.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="The batch size.")
    parser.add_argument("--keep_prob", type=float, default=0.5,
                        help="The keep probability of the dropout layer.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
