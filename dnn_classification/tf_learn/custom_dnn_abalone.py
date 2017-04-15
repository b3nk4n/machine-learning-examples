from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import tempfile
from six.moves import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
LEARNING_RATE = 0.001


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

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def params(self):
        return {self.LEARNING_RATE_KEY: self.learning_rate}

    def build(self, features, targets, mode, params):
        # 1. Configure the model via TensorFlow operations
        # input_layer = tf.contrib.layers.input_from_feature_columns(
        #     columns_to_tensors=features, feature_columns=[age, height, weight])

        # Connect the first hidden layer to input layer (features) with relu activation
        first_hidden_layer = tf.contrib.layers.relu(features, 10)

        # Connect the second hidden layer to first hidden layer with relu
        second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

        # Connect the output layer to second hidden layer (no activation fn)
        output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

        # Reshape output layer to 1-dim Tensor to return predictions
        predictions = tf.reshape(output_layer, [-1])

        # 2. Define the loss function for training/evaluation
        loss = tf.losses.mean_squared_error(targets, predictions)
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.cast(targets, tf.float32), predictions)
        }

        # 3. Define the training operation/optimizer
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=params[self.LEARNING_RATE_KEY],
            optimizer="SGD")

        # 4. Generate predictions
        predictions_dict = {"ages": predictions}

        # 5. Return predictions/loss/train_op/eval_metric_ops in ModelFnOps object
        return model_fn.ModelFnOps(mode, predictions_dict, loss, train_op, eval_metric_ops)


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

    m = CustomModel(LEARNING_RATE)
    nn = tf.contrib.learn.Estimator(model_fn=m.build, params=m.params())

    nn.fit(x=training_set.data, y=training_set.target, steps=5000)

    ev = nn.evaluate(x=test_set.data, y=test_set.target, steps=1)
    print("Loss: {}".format(ev["loss"]))
    print("RMSE: {}".format(ev["rmse"]))

    # Print out predictions
    predictions = nn.predict(x=prediction_set.data, as_iterable=True)
    for i, p in enumerate(predictions):
        print("Prediction %s: %s" % (i + 1, p["ages"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_data", type=str, default="", help="Path to the training data.")
    parser.add_argument(
        "--test_data", type=str, default="", help="Path to the test data.")
    parser.add_argument(
        "--predict_data",
        type=str,
        default="",
        help="Path to the prediction data.")
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
