"""
Trains a Siamese MLP on pairs of digits from the MNIST dataset.
It follows Hadsell-et-al.'06 [1] by computing the Euclidean distance on the
output of the shared network and by optimizing the contrastive loss (see paper
for mode details).
# References
- Dimensionality Reduction by Learning an Invariant Mapping
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
Gets to 97.2% test accuracy after 20 epochs.
2 seconds per epoch on a Titan X Maxwell GPU
"""
from __future__ import absolute_import
from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

DISTANCE_THRESHOLD = 0.52
NUM_CLASSES = 10
SEED = 42


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.keras.backend.sum(tf.square(x - y), axis=1, keepdims=True)
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_square, tf.keras.backend.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def contrastive_loss(y_true, y_pred):
    """Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    margin = 1
    sqaure_pred = tf.keras.backend.square(y_pred)
    margin_square = tf.square(tf.keras.backend.maximum(margin - y_pred, 0))
    return tf.keras.backend.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
    """Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(NUM_CLASSES)]) - 1
    for d in range(NUM_CLASSES):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = np.random.randint(1, NUM_CLASSES)
            dn = (d + inc) % NUM_CLASSES
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_nn_network(input_shape):
    """Base NN network to be shared (eq. to feature extraction).
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.225)(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.225)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-7))(x)
    return tf.keras.models.Model(inputs, x)


def create_base_cnn_network(input_shape):
    """Base CNN network to be shared (eq. to feature extraction).
    """
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.225)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same',
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(0.225)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    return tf.keras.models.Model(inputs, x)


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < DISTANCE_THRESHOLD
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return tf.keras.backend.mean(tf.equal(y_true, tf.cast(y_pred < DISTANCE_THRESHOLD, y_true.dtype)))


def get_digit_indices(labels, examples_per_class):
    digit_indices = [np.where(labels == i)[0] for i in range(NUM_CLASSES)]
    return [di[:examples_per_class] for di in digit_indices]


def plot_values(train_values, valid_values, y_label):
    epochs = range(1, len(train_values) + 1)
    plt.clf()
    plt.plot(epochs, train_values, 'b')
    if valid_values is not None:
        plt.plot(epochs, valid_values, 'g')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.show()


def plot_examples_separated(image_pairs, labels, predictions):
    num = image_pairs.shape[0]
    fig = plt.figure(1)
    for i in range(0, num):
        # works because labels are alternating in unshuffled dataset
        img0 = image_pairs[i, 0][:, :, 0]
        img1 = image_pairs[i, 1][:, :, 0]
        label = labels[i]
        distance = predictions[i, 0]
        fig.add_subplot(num // 2, 4, (i * 2 + 1))
        plt.imshow(img0)
        fig.add_subplot(num // 2, 4, (i * 2 + 2))
        plt.imshow(img1)
        plt.xlabel('==' if label == 0 else '!=')
        plt.ylabel('{:.4f}'.format(distance))
    plt.show()


def plot_examples(image_pairs, labels, predictions):
    num = image_pairs.shape[0]
    fig = plt.figure(1)
    for i in range(0, num):
        img0 = image_pairs[i, 0][:, :, 0]
        img1 = image_pairs[i, 1][:, :, 0]
        label = labels[i]
        distance = predictions[i, 0]
        fig.add_subplot(num, 2, (2*i + 1))
        plt.imshow(img0)
        fig.add_subplot(num, 2, (2*i + 2))
        plt.imshow(img1)
        plt.ylabel('{:.4f}'.format(distance))
    plt.show()


def main(args):
    # results can still be non-deterministic when running on GPU, due to cuDNN
    tf.set_random_seed(SEED)
    np.random.seed(SEED)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1).astype('float32')
    x_test = np.expand_dims(x_test, axis=-1).astype('float32')
    x_train /= 255
    x_test /= 255
    input_shape = x_train.shape[1:]

    # create training+test positive and negative pairs
    tr_digit_indices = get_digit_indices(y_train, args.examples_per_class)
    tr_pairs, tr_y = create_pairs(x_train, tr_digit_indices)

    te_digit_indices = get_digit_indices(y_test, args.examples_per_class)
    te_pairs, te_y = create_pairs(x_test, te_digit_indices)

    # network definition
    if args.base_network == 'nn':
        base_network = create_base_nn_network(input_shape)
    else:
        base_network = create_base_cnn_network(input_shape)

    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)

    # because we re-use the same instance `base_network`,
    # the weights of the network
    # will be shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = tf.keras.layers.Lambda(euclidean_distance,
                                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = tf.keras.models.Model([input_a, input_b], distance)

    model.summary()

    # train
    rms = tf.keras.optimizers.RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    callbacks = []
    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_accuracy'))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, verbose=1))

    if args.min_epochs:
        print('Pre-training...')
        # we pre-train the model (some steps without early stopping), because it takes a while
        # until the accuracy starts to improve
        model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                  batch_size=args.batch_size,
                  epochs=args.min_epochs,
                  verbose=2,
                  validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    print('Training...')
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
                        batch_size=args.batch_size,
                        epochs=args.max_epochs,
                        initial_epoch=args.min_epochs,
                        callbacks=callbacks,
                        verbose=2,
                        validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y))

    plot_values(history.history['loss'], history.history['val_loss'], 'Loss')
    plot_values(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy')

    # compute final accuracy on training and test sets
    y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred)
    y_pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(te_y, y_pred)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # plot first 20 examples
    image_pairs = te_pairs[:20, :]
    labels = te_y[:20]
    predictions = y_pred[:20]

    plot_examples_separated(image_pairs, labels, predictions)

    # plot first 10 FPs
    image_pairs = []
    labels = np.zeros(10)
    predictions = np.zeros((10, 1))
    index = 0
    while len(image_pairs) < 10:
        if y_pred[index] < DISTANCE_THRESHOLD and te_y[index] == 0:
            image_pairs += [[te_pairs[index, 0], te_pairs[index, 1]]]
            labels[len(image_pairs) - 1] = te_y[index]
            predictions[len(image_pairs) - 1, 0] = y_pred[index, 0]
        index += 1
    image_pairs = np.array(image_pairs)

    plot_examples(image_pairs, labels, predictions)

    # plot first 10 FNs
    image_pairs = []
    labels = np.zeros(10)
    predictions = np.zeros((10, 1))
    index = 0
    while len(image_pairs) < 10:
        if y_pred[index] >= DISTANCE_THRESHOLD and te_y[index] == 1:
            image_pairs += [[te_pairs[index, 0], te_pairs[index, 1]]]
            labels[len(image_pairs) - 1] = te_y[index]
            predictions[len(image_pairs) - 1, 0] = y_pred[index, 0]
        index += 1
    image_pairs = np.array(image_pairs)

    plot_examples(image_pairs, labels, predictions)

    # classify (using minimum distance)
    print('Classifying test set...')
    min_correct_counter = 0
    median_correct_counter = 0
    mean_correct_counter = 0
    for t in range(x_test.shape[0]):
        test_img = x_test[t]
        test_img_label = y_test[t]
        n = min([len(tr_digit_indices[d]) for d in range(NUM_CLASSES)]) - 1
        min_aggregated_distance = 999
        min_aggregated_distance_label = -1
        mean_aggregated_distance = 999
        mean_aggregated_distance_label = -1
        median_aggregated_distance = 999
        median_aggregated_distance_label = -1
        for d in range(NUM_CLASSES):
            image_pairs = []
            for i in range(n):
                z1 = tr_digit_indices[d][i]
                img = x_train[z1]
                image_pairs += [[img, test_img]]
            image_pairs = np.array(image_pairs)
            predictions = model.predict([image_pairs[:, 0], image_pairs[:, 1]])

            min_distance = np.min(predictions)
            if min_distance < min_aggregated_distance:
                min_aggregated_distance = min_distance
                min_aggregated_distance_label = d

            median_distance = np.median(predictions)
            if median_distance < median_aggregated_distance:
                median_aggregated_distance = median_distance
                median_aggregated_distance_label = d

            mean_distance = np.mean(predictions)
            if mean_distance < mean_aggregated_distance:
                mean_aggregated_distance = mean_distance
                mean_aggregated_distance_label = d

        if test_img_label == min_aggregated_distance_label:
            min_correct_counter += 1

        if test_img_label == median_aggregated_distance_label:
            median_correct_counter += 1

        if test_img_label == mean_aggregated_distance_label:
            mean_correct_counter += 1

    print('Classification accuracy using MIN:    {}'.format(min_correct_counter / x_test.shape[0]))
    print('Classification accuracy using MEDIAN: {}'.format(median_correct_counter / x_test.shape[0]))
    print('Classification accuracy using MEAN:   {}'.format(mean_correct_counter / x_test.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_epochs', type=int, default=50,
                        help='The minimum number of (pre-)training epochs')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='The maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The batch size while training')
    parser.add_argument('--examples_per_class', type=int, default=10, # TODO create a similar MNIST example where we limit the examples_per_class as a param, to see how a standard classification model performs when the dataset is small
                        help='Maximum number of examples per class')
    parser.add_argument('--base_network', type=str, default='cnn',
                        help='The base network model (nn, cnn) used in the siamese')
    parser.add_argument('--early_stopping', type=bool, default=True,
                        help='Whether to use early stopping or not')
    args = parser.parse_args()
    main(args)
