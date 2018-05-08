import os
import sys
import argparse

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib.keras import backend as K


def preprocess_img(img_path, target_size):
    img = preprocessing.image.load_img(img_path, target_size=target_size)
    x = preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = applications.vgg19.preprocess_input(x)
    return x


def deprocess_img(x):
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def content_loss(base, combination):
    return K.sum(K.square(combination - base))


def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, target_size):
    s = gram_matrix(style)
    c = gram_matrix(combination)
    channels = 3
    img_height, img_width = target_size
    size = img_height * img_width
    return K.sum(K.square(s - c)) / (4.0 * (channels ** 2) * (size ** 2))


def total_variation_loss(x, target_size):
    img_height, img_width = target_size
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


class LossAndGradsCache(object):
    def __init__(self, fetch_loss_and_grads, target_size):
        self.fetch_loss_and_grads = fetch_loss_and_grads
        self.img_height = target_size[0]
        self.img_width = target_size[1]
        self.loss_value = None
        self.grads_value = None

    def loss(self, x):
        assert self.loss_value is None

        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        self.loss_value = outs[0]
        self.grads_value = outs[1].flatten().astype('float64')
        return self.loss_value

    def grads(self, _):
        assert self.loss_value is not None
        grads_values = np.copy(self.grads_value)
        self.loss_value = None
        self.grads_value = None
        return grads_values


def main(_):
    width, height = preprocessing.image.load_img(FLAGS.target_img_path).size
    gen_img_height = 400
    gen_img_width = int(width * gen_img_height / height)

    target_x = preprocess_img(FLAGS.target_img_path, target_size=(gen_img_height, gen_img_width))
    target_img = K.constant(target_x)
    style_x = preprocess_img(FLAGS.style_img_path, target_size=(gen_img_height, gen_img_width))
    style_img = K.constant(style_x)
    combination_img = K.placeholder(shape=(1, gen_img_height, gen_img_width, 3))

    input_tensor = K.concatenate([
        target_img,
        style_img,
        combination_img
    ], axis=0)

    model = applications.vgg19.VGG19(input_tensor=input_tensor,
                                     weights='imagenet',
                                     include_top=False)
    model.summary()

    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    total_variation_weight = 1e-4
    style_weight = 1.0
    content_weight = 0.025

    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_img_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_img_features, combination_features)

    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features,
                        target_size=(gen_img_height, gen_img_width))
        loss += (style_weight / len(style_layers)) * sl

    loss += total_variation_weight * total_variation_loss(combination_img, target_size=(gen_img_height, gen_img_width))

    # setup gradient-descent
    grads_list = K.gradients(loss, combination_img)
    grads = grads_list[0]

    fetch_loss_and_grads = K.function(inputs=[combination_img],
                                      outputs=[loss, grads])

    lossAndGradsCache = LossAndGradsCache(fetch_loss_and_grads,
                                          target_size=(gen_img_height, gen_img_width))

    x = preprocess_img(FLAGS.target_img_path, target_size=(gen_img_height, gen_img_width))
    x = x.flatten()

    for i in range(FLAGS.iterations):
        start_time = time.time()

        x, min_val, info = fmin_l_bfgs_b(lossAndGradsCache.loss,
                                         x,
                                         fprime=lossAndGradsCache.grads,
                                         maxfun=20)
        print('@{:4d}: {:.4f}'.format(i + 1, min_val))

        x_copy = x.copy().reshape((gen_img_height, gen_img_width, 3))
        print(np.min(x_copy), np.mean(x_copy), np.max(x_copy))
        img = deprocess_img(x_copy)
        os.makedirs('out', exist_ok=True)
        filename = 'out/result_{:04d}.png'.format(i + 1)
        imsave(filename, img)
        print('Iteration took {:.1f}s'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20,
                        help='The number of gradient ascent iterations')
    parser.add_argument('--target_img_path', type=str, default='tmp/dogs-vs-cats/train/cats/cat.1.jpg',
                        help='The target image path to apply the style on')
    parser.add_argument('--style_img_path', type=str, default='inputs/irises_public_domain.jpg',
                        help='The style image path to take the style from')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
