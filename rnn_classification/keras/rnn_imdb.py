import argparse
import sys

from tensorflow.contrib.keras import datasets
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import preprocessing

from utils.keras import plots


def create_model(max_features, n_hidden):
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=max_features, output_dim=n_hidden))
    model.add(layers.LSTM(n_hidden))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def main(_):
    (x_train, y_train), (x_test, y_test) = datasets.imdb.load_data(num_words=FLAGS.max_features)
    print('Train sequeces: {} Test sequences: {}'.format(len(x_train), len(x_test)))

    x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=FLAGS.max_seq_len)
    x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=FLAGS.max_seq_len)
    print('Train shape: {} Test shape: {}'.format(x_train.shape, x_test.shape))

    model = create_model(FLAGS.max_features, FLAGS.n_hidden)
    model.compile(optimizer=optimizers.RMSprop(FLAGS.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['acc'])
    result = model.fit(x_train,
                       y_train,
                       epochs=FLAGS.epochs,
                       batch_size=FLAGS.batch_size,
                       validation_split=0.2)

    plots.show_loss(result.history['loss'],
                    result.history['val_loss'])
    plots.show_accuracy(result.history['acc'],
                        result.history['val_acc'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='The initial learning rate')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of training epochs')
    parser.add_argument('--n_hidden', type=int, default=32,
                        help='The number of training epochs')
    parser.add_argument('--max_features', type=int, default=10000,
                        help='The max number of features/words in the dictionary')
    parser.add_argument('--max_seq_len', type=int, default=500,
                        help='The max number of words in a sequence')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
