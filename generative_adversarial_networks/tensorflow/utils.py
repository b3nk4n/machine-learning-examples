import requests
import os
import zipfile

from glob import glob
import numpy as np
from scipy.misc import imread, imsave, imresize
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


def scale_image(image):
    # scale to (-1, +1)
    return (image / 255.0) * 2 - 1


def crop_and_resave(input_file, output_dir):
    # theoretically, we could try to find the face
    # but let's be lazy
    # we assume that the middle 108 pixels will contain the face
    image = imread(input_file)
    height, width, color = image.shape
    edge_h = int(round((height - 108) / 2.0))
    edge_w = int(round((width - 108) / 2.0))

    cropped = image[edge_h:(edge_h + 108), edge_w:(edge_w + 108)]
    small = imresize(cropped, (64, 64))

    filename = input_file.split('/')[-1]
    imsave("%s/%s" % (output_dir, filename), small)


def files2images(filenames):
    return [scale_image(imread(fn)) for fn in filenames]


def download_file(file_id, dest):
    drive_url = "https://docs.google.com/uc?export=download"

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    session = requests.Session()
    response = session.get(drive_url, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(drive_url, params=params, stream=True)

    def save_response_content(r, dest):
        # unfortunately content-length is not provided in header
        total_iters = 1409659  # in KB
        print("Note: units are in KB, e.g. KKB = MB")
        # because we are reading 1024 bytes at a time, hence
        # 1KB == 1 "unit" for tqdm
        with open(dest, 'wb') as f:
            for chunk in tqdm(
                    r.iter_content(1024),
                    total=total_iters,
                    unit='KB',
                    unit_scale=True):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    save_response_content(response, dest)


def get_celeb():
    if not os.path.exists('../large_files'):
        os.mkdir('../large_files')

    # eventual place where our final data will reside
    if not os.path.exists('../large_files/img_align_celeba-cropped'):

        # check for original data
        if not os.path.exists('../large_files/img_align_celeba'):
            # download the file and place it here
            if not os.path.exists('../large_files/img_align_celeba.zip'):
                print("Downloading img_align_celeba.zip...")
                download_file(
                    '0B7EVK8r0v71pZjFTYXZWM3FlRnM',
                    '../large_files/img_align_celeba.zip'
                )

            # unzip the file
            print("Extracting img_align_celeba.zip...")
            with zipfile.ZipFile('../large_files/img_align_celeba.zip') as zf:
                zf.extractall('../large_files')

        # load in the original images
        filenames = glob("../large_files/img_align_celeba/*.jpg")
        n = len(filenames)
        print("Found %d files!" % n)

        # crop the images to 64x64
        os.mkdir('../large_files/img_align_celeba-cropped')
        print("Cropping images, please wait...")

        for i in range(n):
            crop_and_resave(filenames[i], '../large_files/img_align_celeba-cropped')
            if i % 1000 == 0:
                print("%d/%d" % (i, n))

    # make sure to return the cropped version
    filenames = glob("../large_files/img_align_celeba-cropped/*.jpg")
    return filenames


def get_mnist():
    mnist = input_data.read_data_sets('../../data/tmp/mnist', one_hot=False)
    Xtrain, Ytrain = mnist.train.images, mnist.train.labels
    Xtrain = Xtrain.astype(np.float32)
    return Xtrain, Ytrain
