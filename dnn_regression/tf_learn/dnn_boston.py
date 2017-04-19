from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import argparse
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"
COLUMNS = FEATURES + [LABEL]


def input_fn(data_set):
    feature_cols_dict = {f: tf.constant(data_set[f].values, shape=[data_set[f].values.shape[0], 1]) for f in FEATURES}
    labels = tf.constant(data_set[LABEL].values)
    return feature_cols_dict, labels


def main(_):
    training_set = pd.read_csv("../../data/boston/boston_train.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)
    test_set = pd.read_csv("../../data/boston/boston_test.csv", skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
    prediction_set = pd.read_csv("../../data/boston/boston_predict.csv", skipinitialspace=True,
                                 skiprows=1, names=COLUMNS)

    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
                                              hidden_units=[10, 10],
                                              model_dir="tmp/boston_model",
                                              activation_fn=tf.nn.relu,
                                              dropout=0.5,
                                              optimizer="Adam")

    print("Training...")
    regressor.fit(input_fn=lambda: input_fn(training_set), steps=FLAGS.train_steps)

    print("Eval...")
    ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
    print("Evaluation: ", ev)

    loss_score = ev["loss"]
    print("Loss: {0:f}".format(loss_score))

    print("Predictions...")
    predict_result_list = regressor.predict(input_fn=lambda: input_fn(prediction_set))

    predictions = list(predict_result_list)
    print ("Predictions: {}".format(str(predictions)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=1000,
                        help='The number of training steps')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
