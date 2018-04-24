import argparse
import sys

from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import optimizers

import cnn_classification.keras.dogs_cats_dataset as dataset
import cnn_classification.keras.utils as utils


def create_model(dropout_rate):
    model = models.Sequential()
    conv_base = applications.VGG16(include_top=False,
                                   input_shape=(150, 150, 3),
                                   weights='imagenet')
    conv_base.trainable = False
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def main(_):
    train_dir, valid_dir, test_dir = dataset.prepare()

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

    train_datagen = dataset.get_generator(train_dir, FLAGS.batch_size, augmentation=FLAGS.augmentation)
    valid_datagen = dataset.get_generator(valid_dir, FLAGS.batch_size, augmentation=False)
    test_datagen = dataset.get_generator(test_dir, FLAGS.batch_size, augmentation=False)

    res = model.fit_generator(
        train_datagen,
        steps_per_epoch=num_train // FLAGS.batch_size,
        epochs=FLAGS.epochs,
        validation_data=valid_datagen,
        validation_steps=num_valid // FLAGS.batch_size)

    model.save('cats_and_dogs_vgg16_{}.h5'.format(num_train))

    utils.show_accuracy(res.history['acc'],
                        res.history['val_acc'])
    utils.show_loss(res.history['loss'],
                    res.history['val_loss'])

    scores = model.evaluate_generator(
        test_datagen,
        steps=num_test // FLAGS.batch_size)

    print('Test result:')
    print('Loss: {} Accuracy: {}'.format(scores[0], scores[1]))

    if FLAGS.fine_tuning_epochs > 0:
        # make block5 trainable for fine-tuning
        for layer in model.layers[0].layers:
            if layer.name.startswith('block5_conv'):
                layer.trainable = True
        model.summary()

        model.compile(optimizer=optimizers.Adam(FLAGS.learning_rate / 2),
                      loss='binary_crossentropy',
                      metrics=['acc'])

        res = model.fit_generator(
            train_datagen,
            steps_per_epoch=num_train // FLAGS.batch_size,
            epochs=FLAGS.fine_tuning_epochs,
            validation_data=valid_datagen,
            validation_steps=num_valid // FLAGS.batch_size)

        model.save('cats_and_dogs_vgg16_finetuned_{}.h5'.format(num_train))

        utils.show_accuracy(res.history['acc'],
                            res.history['val_acc'])
        utils.show_loss(res.history['loss'],
                        res.history['val_loss'])

        scores = model.evaluate_generator(
            test_datagen,
            steps=num_test // FLAGS.batch_size)

        print('Test result after fine-tuning:')
        print('Loss: {} Accuracy: {}'.format(scores[0], scores[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='The initial learning rate')
    parser.add_argument('--epochs', type=int, default=1,
                        help='The number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='The dropout used after the conv-stack')
    parser.add_argument('--augmentation', type=bool, default=True,
                        help='Whether to use data augmentation')
    parser.add_argument('--fine_tuning_epochs', type=int, default=2,
                        help='Do fine-tuning of block5 in case non-zero.')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
