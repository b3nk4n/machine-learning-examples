import argparse
import os
import sys
from os.path import expanduser
import cnn_classification.keras.utils as utils
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import optimizers
from tensorflow.contrib.keras import preprocessing


SMALL_DATASET_PATH = '../../data/dogs-vs-cats'
DATA_PATH = os.path.join(expanduser("~"), '.kaggle/competitions/dogs-vs-cats')
TRAIN_DIR = 'train'
TEST_DIR = 'test1'


def prepare_dataset(train_size=2000, valid_size=1000, test_size=1000):
    """
    Use a small dataset by default to show overfitting effects.
    Notes: We lazily ignore the test1 folder, because the data in there is not labeled.
           The full dataset would have 25k, not just 12.5k examples.
    """
    assert train_size + valid_size + test_size <= 12500

    full_train_root = utils.extract_zipfile(DATA_PATH, TRAIN_DIR)

    base_dir = SMALL_DATASET_PATH

    start = 0
    train_dir = os.path.join(base_dir, 'train')
    if not os.path.isdir(train_dir):
        utils.ensure_directory(base_dir, 'train')
        train_cats_dir = utils.ensure_directory(train_dir, 'cats')
        train_dogs_dir = utils.ensure_directory(train_dir, 'dogs')

        train_size_per_class = train_size // 2
        utils.copy_files('cat.{}.jpg', start, train_size_per_class, full_train_root, train_cats_dir)
        utils.copy_files('dog.{}.jpg', start, train_size_per_class, full_train_root, train_dogs_dir)
        start += train_size_per_class

    valid_dir = os.path.join(base_dir, 'valid')
    if not os.path.isdir(valid_dir):
        utils.ensure_directory(base_dir, 'valid')
        valid_cats_dir = utils.ensure_directory(valid_dir, 'cats')
        valid_dogs_dir = utils.ensure_directory(valid_dir, 'dogs')

        valid_size_per_class = valid_size // 2
        utils.copy_files('cat.{}.jpg', start, valid_size_per_class, full_train_root, valid_cats_dir)
        utils.copy_files('dog.{}.jpg', start, valid_size_per_class, full_train_root, valid_dogs_dir)
        start += valid_size_per_class

    test_dir = os.path.join(base_dir, 'test')
    if not os.path.isdir(test_dir):
        utils.ensure_directory(base_dir, 'test')
        test_cats_dir = utils.ensure_directory(test_dir, 'cats')
        test_dogs_dir = utils.ensure_directory(test_dir, 'dogs')

        test_size_per_class = test_size // 2
        utils.copy_files('cat.{}.jpg', start, test_size_per_class, full_train_root, test_cats_dir)
        utils.copy_files('dog.{}.jpg', start, test_size_per_class, full_train_root, test_dogs_dir)

    return train_dir, valid_dir, test_dir


def create_data_generator(data_dir, batch_size, augmentation):
    if augmentation:
        generator = preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    else:
        generator = preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    datagen = generator.flow_from_directory(
        data_dir,
        (150, 150),
        batch_size=batch_size,
        class_mode='binary')
    return datagen


def create_model(dropout_rate):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def main(_):
    train_dir, valid_dir, test_dir = prepare_dataset()

    num_train = len(utils.listdir(train_dir, recursive=True))
    num_valid = len(utils.listdir(valid_dir, recursive=True))
    num_test = len(utils.listdir(test_dir, recursive=True))

    print('Training images:   {:5}'.format(num_train))
    print('Validation images: {:5}'.format(num_valid))
    print('Test images:       {:5}'.format(num_test))

    model = create_model(FLAGS.dropout)
    model.summary()
    model.compile(optimizer=optimizers.Adam(FLAGS.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    train_datagen = create_data_generator(train_dir, FLAGS.batch_size, augmentation=FLAGS.augmentation)
    valid_datagen = create_data_generator(valid_dir, FLAGS.batch_size, augmentation=False)
    test_datagen = create_data_generator(test_dir, FLAGS.batch_size, augmentation=False)

    res = model.fit_generator(
        train_datagen,
        steps_per_epoch=num_train // FLAGS.batch_size,
        epochs=FLAGS.epochs,
        validation_data=valid_datagen,
        validation_steps=num_valid // FLAGS.batch_size)

    model.save('cats_and_dogs_{}.h5'.format(num_train))

    utils.show_accuracy(res.history['acc'],
                        res.history['val_acc'])
    utils.show_loss(res.history['loss'],
                    res.history['val_loss'])

    scores = model.evaluate_generator(
        test_datagen,
        steps=num_test // FLAGS.batch_size)

    print('Test result:')
    print('Loss: {} Accuracy: {}'.format(scores[0], scores[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='The initial learning rate')
    parser.add_argument('--epochs', type=int, default=30,
                        help='The number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='The dropout used after the conv-stack')
    parser.add_argument('--augmentation', type=bool, default=True,
                        help='Whether to use data augmentation')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
