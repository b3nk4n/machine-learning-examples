import os
from os.path import expanduser

from tensorflow.contrib.keras import preprocessing

import cnn_classification.keras.utils as utils

SMALL_DATASET_PATH = 'tmp/dogs-vs-cats'
DATA_PATH = os.path.join(expanduser("~"), '.kaggle/competitions/dogs-vs-cats')
TRAIN_DIR = 'train'
TEST_DIR = 'test1'


def prepare(train_size=2000, valid_size=1000, test_size=1000):
    """
    Use a small dataset by default to show overfitting effects.
    Notes: We lazily ignore the test1 folder, because the data in there is not labeled.
           The full dataset would have 25k, not just 12.5k examples.
    """
    assert train_size + valid_size + test_size <= 12500

    full_train_root = utils.extract_zipfile(DATA_PATH, TRAIN_DIR)

    base_dir = SMALL_DATASET_PATH
    os.makedirs(base_dir, exist_ok=True)

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


def get_generator(data_dir, batch_size, augmentation):
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
