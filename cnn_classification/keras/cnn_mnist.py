import argparse
import numpy as np
import tensorflow as tf

NUM_CLASSES = 10


def get_digit_indices(labels, examples_per_class):
    digit_indices = [np.where(labels == i)[0] for i in range(NUM_CLASSES)]
    return [di[:examples_per_class] for di in digit_indices]


def main(args):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # limit training set to simulate overfitting, and to compare to one_shot_learning/keras/siamese_mnist.py
    tr_digit_indices = get_digit_indices(y_train, args.examples_per_class)

    x_train_limited = np.zeros((args.examples_per_class * NUM_CLASSES, 28, 28))
    y_train_limited = np.zeros(args.examples_per_class * NUM_CLASSES)
    index = 0
    for d in range(NUM_CLASSES):
        for i in range(args.examples_per_class):
            x_train_limited[index] = x_train[tr_digit_indices[d][i]]
            y_train_limited[index] = y_train[tr_digit_indices[d][i]]
            index += 1

    if args.model == 'nn':
        model = create_nn_model()
    else:
        model = create_cnn_model()

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train_limited, y_train_limited,
              epochs=args.epochs,
              batch_size=args.batch_size,
              verbose=2,
              validation_data=(x_test, y_test))
    model.evaluate(x_test, y_test, verbose=2)


def create_nn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(5e-6)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.225),
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(5e-6)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.225),
        tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(5e-6))
    ])
    return model


def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.225),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.225),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(5e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(5e-4))
    ])
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25,
                        help='The number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size while training')
    parser.add_argument('--examples_per_class', type=int, default=25,
                        help='Maximum number of examples per class')
    parser.add_argument('--model', type=str, default='cnn',
                        help='The network model (nn, cnn) used')
    args = parser.parse_args()
    main(args)
