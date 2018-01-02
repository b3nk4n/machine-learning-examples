import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from hyperparam_opt.tensorflow.dnn_mnist import HyperParams, train
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
from tensorflow.examples.tutorials.mnist import input_data


def optimizer(args):
    hyper = HyperParams(**args)
    print(hyper.to_string())

    # read a new dataset every time, because there is no reset() function
    mnist = input_data.read_data_sets(FLAGS.data_dir)

    # make a new logs-subdir for every run
    timestamp = int(time.time())
    log_dir = os.path.join(FLAGS.log_dir, str(timestamp))

    loss, accuracy, epochs = train(mnist, hyper, log_dir)
    return {
            'status': STATUS_OK,
            'loss': loss,
            'epochs': epochs,
            'metrics': {
                'accuracy': accuracy
            }
        }


def main(_):
    space = {
        'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
        'batch_size': hp.quniform('batch_size', 8, 256, 2),
        'n_hidden': hp.quniform('n_hidden', 32, 256, 1),
        'keep_prob': hp.uniform('keep_prob', 0.2, 1.0),
    }

    t = Trials()
    best = fmin(optimizer, space, algo=tpe.suggest, max_evals=10, trials=t)
    print('TPE best: {}'.format(best))

    for trial in t.trials:
        print('{} --> {}'.format(trial['result'], trial['misc']['vals']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for storing logs')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
