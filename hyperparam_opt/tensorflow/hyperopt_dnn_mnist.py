import sys
import argparse
import tensorflow as tf

from hyperparam_opt.tensorflow.dnn_mnist import HyperParams, train
from hyperopt import fmin, hp, Trials, tpe, STATUS_OK
from tensorflow.examples.tutorials.mnist import input_data

MNIST = None


def optimizer(args):
    hyper = HyperParams(**args)
    print(hyper.to_string())

    loss, accuracy, epochs = train(MNIST, hyper)
    return {
            'status': STATUS_OK,
            'loss': loss,
            'epochs': epochs,
            'metrics': {
                'accuracy': accuracy
            }
        }


def main(_):
    # Import data
    global MNIST
    MNIST = input_data.read_data_sets(FLAGS.data_dir)

    space = {
        'lr': hp.uniform('lr', 0.0001, 0.01),
        'batch_size': hp.quniform('batch_size', 8, 256, 2),
        'n_hidden': hp.quniform('n_hidden', 32, 256, 1),
        'keep_prob': hp.uniform('keep_prob', 0.2, 1.0),
    }

    t = Trials()
    best = fmin(optimizer, space, algo=tpe.suggest, max_evals=10, trials=t)
    print('TPE best:'.format(best))

    for trial in t.trials:
        print('{} --> {}'.format(trial['result'], trial['misc']['vals']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/tmp/mnist',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
