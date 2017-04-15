from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

NAMED_COLUMNS = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]


def main():
    # If the training and test sets aren't stored locally, download them.
    if not os.path.exists(IRIS_TRAINING):
        raw = urllib.urlopen(IRIS_TRAINING_URL).read()
        with open(IRIS_TRAINING, "w") as f:
            f.write(raw)

    if not os.path.exists(IRIS_TEST):
        raw = urllib.urlopen(IRIS_TEST_URL).read()
        with open(IRIS_TEST, "w") as f:
            f.write(raw)

    # Load data sets
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=IRIS_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Define the training inputs
    def get_inputs(data_set):
        print(type(data_set))
        x = {n: tf.constant(data_set.data[:, i], shape=[data_set.data.shape[0], 1])
             for i, n in enumerate(NAMED_COLUMNS)}
        y = tf.constant(data_set.target)

        for j in NAMED_COLUMNS:
            print("x", x[j].get_shape().as_list())
        print("y", y.get_shape().as_list())

        return x, y

    # Specify that all features have real-value data
    feature_columns = [tf.contrib.layers.real_valued_column(k) for k in NAMED_COLUMNS]

    validation_metrics = {
        "accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_accuracy,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_precision,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES),
        "recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.contrib.metrics.streaming_recall,
                prediction_key=tf.contrib.learn.PredictionKey.CLASSES)
    }

    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        input_fn=lambda: get_inputs(test_set),
        eval_steps=1,
        every_n_steps=50,
        metrics=validation_metrics,
        early_stopping_metric="accuracy",
        early_stopping_metric_minimize=True,
        early_stopping_rounds="100")

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[16, 32, 16],
                                                n_classes=3,
                                                model_dir="tmp/iris_model",
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1))

    # Fit model.
    classifier.fit(input_fn=lambda: get_inputs(training_set),
                   steps=5000,
                   monitors=[validation_monitor])

    # Evaluate accuracy.
    ev = classifier.evaluate(input_fn=lambda: get_inputs(test_set), steps=1)

    print("Evaluation: ", ev)

    # Classify two new flower samples.
    def new_samples():
        return {
            "Sepal Length": tf.constant([[6.4], [5.8]]),
            "Petal Length": tf.constant([3.2, 3.1], shape=[2, 1]),
            "Sepal Width": tf.constant([4.5, 5.0], shape=[2, 1]),
            "Petal Width": tf.constant([1.5, 1.7], shape=[2, 1])
        }

    predictions = list(classifier.predict(input_fn=new_samples))

    print("Predictions: {}\n".format(predictions))

if __name__ == "__main__":
    main()
