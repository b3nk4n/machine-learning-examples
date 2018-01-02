import sacred
from sacred.stflow import LogFileWriter

from labwatch.assistant import LabAssistant
from labwatch.optimizers import RandomSearch
from labwatch.hyperparameters import UniformFloat, UniformNumber

from tensorflow.examples.tutorials.mnist import input_data

from hyperparam_opt.tensorflow.dnn_mnist import HyperParams, train

ex = sacred.Experiment(name='MINST')
a = LabAssistant(ex,
                 database_name='labwatch-db',
                 optimizer=RandomSearch)


@ex.config
def cfg():
    lr = 0.001
    batch_size = 100
    n_hidden = 32
    keep_prob = 0.5
    data_dir = '../../data/tmp/mnist/'
    log_dir = './logs/'


@a.search_space
def search_space():
    lr = UniformFloat(0.0001, 0.01, default=0.001, log_scale=True)
    batch_size = UniformNumber(8, 256, default=100, type=int)
    n_hidden = UniformNumber(32, 256, default=32, type=int)
    keep_prob = UniformFloat(0.2, 1.0, default=0.5)


@ex.automain
@LogFileWriter(ex)
def main(lr, batch_size, n_hidden, keep_prob, data_dir, log_dir, _run):
    mnist = input_data.read_data_sets(data_dir)

    hyper = HyperParams(lr, batch_size, n_hidden, keep_prob)
    loss, accuracy, epochs = train(mnist, hyper, log_dir)

    return accuracy

    # Labwatch allows to return scalars or dictionaries (where 'optimization_target' is an required key).
    # Unfortunately, dictionaries are not fully supported
    # results = {
    #     'optimization_target': loss,
    #     'accuracy': accuracy,
    #     'epochs': epochs
    # }
