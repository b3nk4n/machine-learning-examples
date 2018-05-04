import argparse
import random
import sys

import numpy as np
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import utils

GEN_SEQ_LENGTH = 400
MAX_SEQ_LENGTH = 60
CHAR_STEP = 3


def reweight_distribution(src_distribution, temperature=0.5):
    """Reweight distribution to different temperature.

    :param src_distribution: The source distribution
    :param temperature: Quantifies the entropy of the output distribution.
    :return: The re-weighted distribution.
    """
    distribution = np.log(src_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


def sample_next_character(preds, temperature=1.0):
    preds = np.asarray(preds).astype(np.float64)
    preds = reweight_distribution(preds, temperature)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


def create_model(n_hidden, seq_length, num_classes):
    model = models.Sequential()
    model.add(layers.LSTM(n_hidden, input_shape=(seq_length, num_classes)))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


def _vectorize_char_sequeces(text):
    corpus_length = len(text)

    sentences = []
    next_chars = []

    for i in range(0, corpus_length - MAX_SEQ_LENGTH, CHAR_STEP):
        target_idx = i + MAX_SEQ_LENGTH
        sentences.append(text[i:target_idx])
        next_chars.append(text[target_idx])

    num_sentences = len(sentences)
    print('Number of sentences: {}'.format(num_sentences))

    chars = sorted(list(set(text)))
    num_classes = len(chars)
    print('Unique chars/classes: {}'.format(num_classes))
    char_indices = dict([(char, chars.index(char)) for char in chars])

    # vectorization
    x = np.zeros((num_sentences, MAX_SEQ_LENGTH, num_classes), dtype=np.bool)
    y = np.zeros((num_sentences, num_classes), dtype=np.bool)

    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            x[i, j, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return x, y, chars, char_indices


def nietzsche_dataset():
    print('Downloading nietzsche.txt...')
    path = utils.get_file('nietzsche.txt',
                          'https://s3.amazonaws.com/text-datasets/nieztsche.txt')
    text = open(path).read()
    text = text.lower()

    return _vectorize_char_sequeces(text), text


def main(_):
    (x, y, chars, char_indices), text = nietzsche_dataset()

    corpus_length = len(text)
    print('Corpus length: {}'.format(corpus_length))

    num_classes = y.shape[-1]
    model = create_model(FLAGS.num_hidden, MAX_SEQ_LENGTH, num_classes)

    model.compile(optimizer=optimizers.RMSprop(lr=FLAGS.learning_rate),
                  loss='categorical_crossentropy')

    for epoch in range(FLAGS.epochs):
        print('Epoch: {}'.format(epoch + 1))

        model.fit(x, y, batch_size=FLAGS.batch_size, epochs=1)

        start_index = random.randint(0, corpus_length - MAX_SEQ_LENGTH - 1)
        generated_text = text[start_index:start_index + MAX_SEQ_LENGTH]

        print('Generating with seed: {}'.format(generated_text))

        for temperature in [0.2, 0.5, 0.75, 1.0, 1.2]:
            print('Temperature: {}', temperature)
            sys.stdout.write(generated_text)  # TODO why stdout and not just print?

            for i in range(GEN_SEQ_LENGTH):
                sampled = np.zeros((1, MAX_SEQ_LENGTH, num_classes))
                for j, char in enumerate(generated_text):
                    sampled[0, j, char_indices[char]] = 1.0

                preds = model.predict(sampled, verbose=0)
                preds = preds[0]
                next_index = sample_next_character(preds, temperature)
                next_char = chars[next_index]

                generated_text += next_char
                generated_text = generated_text[1:]

                sys.stdout.write(next_char)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60,
                        help='The number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='The learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The batch size')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
