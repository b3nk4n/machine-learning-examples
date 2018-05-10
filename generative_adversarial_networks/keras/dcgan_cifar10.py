import argparse

from tensorflow.contrib.keras import datasets
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib.keras import optimizers

import os
import numpy as np

HEIGHT = 32
WIDTH = 32
CHANNELS = 3

SAVE_DIR = 'tmp'


def cifar10_frogs():
    (x_train, y_train), (_, _) = datasets.cifar10.load_data()
    frogs_class = 6
    x_train = x_train[y_train.flatten() == frogs_class]
    x_train = x_train.reshape((x_train.shape[0], HEIGHT, WIDTH, CHANNELS))
    x_train = x_train.astype(np.float32) / 255
    return x_train


def save_image(filename, x):
    img = preprocessing.image.array_to_img(x * 255.0, scale=False)
    img.save(os.path.join(SAVE_DIR, filename))


class DCGAN(object):
    def __init__(self, args):
        self._generator = None
        self._discriminator = None
        self._adversarial = None
        self.args = args

    def build(self):
        self._generator = self._create_generator()

        self._discriminator = self._create_discriminator()

        self._discriminator.compile(optimizer=optimizers.RMSprop(lr=self.args.d_lr,
                                                                 clipvalue=1.0,
                                                                 decay=1e-8),
                                    loss='binary_crossentropy')

        self._adversarial = self._create_gan()

        self._adversarial.compile(optimizer=optimizers.RMSprop(lr=self.args.gan_lr,
                                                               clipvalue=1.0,
                                                               decay=1e-8),
                                  loss='binary_crossentropy')

    def _create_generator(self):
        inputs = layers.Input(shape=(self.args.latent_dims,))

        x = layers.Dense(128 * 16 * 16)(inputs)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((16, 16, 128))(x)

        x = layers.Conv2D(256, kernel_size=5, strides=1, padding='same')(x)
        x = layers.LeakyReLU()(x)

        # we use a kernel-size which is a multiple of the strides to don't have artifacts when up-sampling
        x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(256, kernel_size=5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(256, kernel_size=5, padding='same')(x)
        x = layers.LeakyReLU()(x)

        outputs = layers.Conv2D(CHANNELS, kernel_size=7, activation='tanh', padding='same')(x)

        generator = models.Model(inputs, outputs)
        return generator

    def _create_discriminator(self):
        inputs = layers.Input(shape=(HEIGHT, WIDTH, CHANNELS))

        x = layers.Conv2D(128, kernel_size=3)(inputs)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, kernel_size=4, strides=2)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, kernel_size=4, strides=2)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Conv2D(128, kernel_size=4, strides=2)(x)
        x = layers.LeakyReLU()(x)

        x = layers.Flatten()(x)
        x = layers.Dropout(self.args.dropout)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        discriminator = models.Model(inputs, outputs)
        return discriminator

    def _create_gan(self):
        # this might cause a wrong WARNING (https://github.com/keras-team/keras/issues/8585)
        self._discriminator.trainable = False

        gan_input = layers.Input(shape=(self.args.latent_dims,))
        gan_output = self._discriminator(self._generator(gan_input))

        gan = models.Model(gan_input, gan_output)
        return gan

    def generate(self):
        random_latent = np.random.normal(size=(self.args.batch_size,
                                               self.args.latent_dims))

        generated_images = self.generator.predict(random_latent)
        return generated_images

    def discriminate(self, generated_images, real_images):
        combined_images = np.concatenate([generated_images, real_images])
        labels = np.concatenate([np.ones((args.batch_size, 1)),
                                 np.zeros((args.batch_size, 1))])
        # trick: add slight random noise to the labels
        labels += 0.05 * np.random.random(labels.shape)
        loss = self.discriminator.train_on_batch(combined_images, labels)
        return loss

    def train_on_batch(self):
        random_latent = np.random.normal(size=(self.args.batch_size,
                                               self.args.latent_dims))

        # pretend everything is real, because D is frozen and G should be trained
        # so that the final GAN output loss is lower (mean lower probability of fake images)
        misleading_targets = np.zeros((self.args.batch_size, 1))

        loss = self.adversarial.train_on_batch(random_latent, misleading_targets)
        return loss

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.adversarial.save_weights(os.path.join(path, 'gan.h5'))

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    @property
    def generator(self):
        return self._generator

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def adversarial(self):
        return self._adversarial


def main(args):
    x_train = cifar10_frogs()

    dcgan = DCGAN(args)
    dcgan.build()

    start = 0
    for step in range(args.steps):
        generated_images = dcgan.generate()

        print(generated_images.min(), generated_images.mean(), generated_images.max())

        end = start + args.batch_size
        real_images = x_train[start:end]

        d_loss = dcgan.discriminate(generated_images, real_images)

        adv_loss = dcgan.train_on_batch()

        start += args.batch_size

        if start > len(x_train) - args.batch_size:
            perm = np.random.permutation(len(x_train))
            x_train = x_train[perm]
            start = 0

        if step % 100 == 0:
            dcgan.save(SAVE_DIR)

            print('D loss: {:.4f}   ADV loss: {}'.format(d_loss, adv_loss))

            save_image('generated_{:05d}.png'.format(step), generated_images[0])
            save_image('real_{:05d}.png'.format(step), real_images[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=10000,
                        help='The number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='The batch size while training')
    parser.add_argument('--gan_lr', type=float, default=0.0004,
                        help='The generators learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0008,
                        help='The discriminators learning rate')
    parser.add_argument('--latent_dims', type=int, default=32,
                        help='The dimensions of the latent space')
    parser.add_argument('--dropout', type=float, default=0.25,
                        help='The dropout in the discriminator to use')
    args = parser.parse_args()
    main(args)
