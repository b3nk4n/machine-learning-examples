import argparse
import os
import sys
import zipfile

from glob import glob

import tensorflow as tf

import generative_adversarial_networks.tensorflow.models as models
import generative_adversarial_networks.tensorflow.utils as utils

DATA_ROOT = '../../data/tmp/celeb'
OUTPUT_ROOT = 'tmp/celeb'


def get_celeb_filenames():
    if not os.path.exists(DATA_ROOT):
        os.mkdir(DATA_ROOT)

    # eventual place where our final data will reside
    if not os.path.exists(os.path.join(DATA_ROOT, 'img_align_celeba-cropped')):

        # check for original data
        if not os.path.exists(os.path.join(DATA_ROOT, 'img_align_celeba')):
            # download the file and place it here
            if not os.path.exists(os.path.join(DATA_ROOT, 'img_align_celeba.zip')):
                print("Downloading img_align_celeba.zip...")
                utils.download_file_from_google_drive(
                    '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
                    os.path.join(DATA_ROOT, 'img_align_celeba.zip')
                )

            # unzip the file
            print("Extracting img_align_celeba.zip...")
            with zipfile.ZipFile(os.path.join(DATA_ROOT, 'img_align_celeba.zip')) as zf:
                zf.extractall(DATA_ROOT)

        # load in the original images
        filenames = glob(os.path.join(DATA_ROOT, "img_align_celeba/*.jpg"))
        n = len(filenames)
        print("Found %d files!" % n)

        # crop the images to 64x64
        os.mkdir(os.path.join(DATA_ROOT, 'img_align_celeba-cropped'))
        print("Cropping images, please wait...")

        for i in range(n):
            utils.crop_and_resave(filenames[i], os.path.join(DATA_ROOT, 'img_align_celeba-cropped'))
            if i % 1000 == 0:
                print("%d/%d" % (i, n))

    # make sure to return the cropped version
    filenames = glob(os.path.join(DATA_ROOT, "img_align_celeba-cropped/*.jpg"))
    return filenames


def main(_):
    # make dir to save samples
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    X = get_celeb_filenames()
    dim = 64
    colors = 3

    d_sizes = {
        'conv_layers': [
            (64, 5, 2, False),
            (128, 5, 2, True),
            (256, 5, 2, True),
            (512, 5, 2, True)
        ],
        'dense_layers': [],
    }
    g_sizes = {
        'z': 100,
        'projection': 512,
        'bn_after_projection': True,
        'conv_layers': [
            (256, 5, 2, True),
            (128, 5, 2, True),
            (64, 5, 2, True),
            (colors, 5, 2, False)
        ],
        'dense_layers': [],
        'output_activation': tf.tanh,
    }

    model = models.DCGAN(dim, colors, d_sizes, g_sizes, FLAGS.lr, FLAGS.beta1)
    model.fit(X, epochs=FLAGS.epochs, batch_size=FLAGS.batch_size,
              save_sample_interval=FLAGS.save_sample_interval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='The batch size while training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='The learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='The beta1 coefficient for the optimizer')
    parser.add_argument('--save_sample_interval', type=int, default=50,
                        help='The interval for saving sample images')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
