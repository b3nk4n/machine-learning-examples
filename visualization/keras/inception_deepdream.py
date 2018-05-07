import argparse
import os
import sys

import numpy as np
import scipy
from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import preprocessing

NUM_OCTAVES = 3
OCTAVES_SCLAE = 1.4
MAX_LOSS = 10.0


def resize_img(img, shape):
    img = np.copy(img)
    print(shape)
    factors = (1,
               float(shape[0]) / img.shape[1],
               float(shape[1]) / img.shape[2],
               1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, filename):
    img = np.copy(img)
    pil_img = deprocess_img(img)
    os.makedirs('output', exist_ok=True)
    out_path = os.path.join('output', filename)
    scipy.misc.imsave(out_path, pil_img)


def preprocess_img(img_path):
    img = preprocessing.image.load_img(img_path)
    img = preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    x = applications.inception_v3.preprocess_input(img)
    return x


def deprocess_img(x):
    x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.0
    x += 0.5
    x *= 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def main(_):
    # disable all training specific operations
    K.set_learning_phase(0)

    model = applications.inception_v3.InceptionV3(weights='imagenet',
                                                  include_top=False)
    layer_contributions = {
        'mixed2': 0.2,
        'mixed3': 3.0,
        'mixed4': 2.0,
        'mixed5': 1.5
    }

    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    loss = K.variable(0.,)
    for layer_name in layer_contributions:
        coeff = layer_contributions[layer_name]
        activation = layer_dict[layer_name].output

        scaling = K.prod(K.cast(K.shape(activation), 'float32'))
        # avoid artifacts by only involving non-boarder pixels
        loss += coeff * K.sum(K.square(activation[:, 2:-2, 2:-2, :])) / scaling

    # start the gradient-ascent process
    dream = model.input

    grads_list = K.gradients(loss, dream)
    grads = grads_list[0]

    # trick: normalize gradients
    grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

    fetch_loss_and_grads = K.function(inputs=[dream],
                                      outputs=[loss, grads])

    def gradient_ascent(x, iterations, step_rate, max_loss=None):
        for i in range(iterations):
            loss_value, grads_value = fetch_loss_and_grads([x])
            if max_loss is not None and loss_value > max_loss:
                break
            print('@{:4d}: {:.4f}'.format(i, loss_value))
            x += step_rate * grads_value
        return x

    img = preprocess_img(FLAGS.img_path)

    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, NUM_OCTAVES):
        shape = tuple([int(dim / (OCTAVES_SCLAE ** i))
                      for dim in original_shape])
        successive_shapes.append(shape)

    # reverse
    successive_shapes = successive_shapes[::-1]

    original_img = np.copy(img)
    shrunk_original_img = resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        print('Preprocess image with shape: {}'.format(shape))
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              iterations=FLAGS.iterations,
                              step_rate=FLAGS.step_rate,
                              max_loss=MAX_LOSS)

        same_size_original = resize_img(original_img, shape)

        if FLAGS.repair_lost_detail:
            upscale_shrunk_original_img = resize_img(shrunk_original_img, shape)
            lost_detail = same_size_original - upscale_shrunk_original_img
            img += lost_detail

        shrunk_original_img = same_size_original
        save_img(img, filename='dream_at_scale_{}.png'.format(str(shape)))

    save_img(img, filename='dream.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=50,
                        help='The number of gradient ascent iterations')
    parser.add_argument('--step_rate', type=float, default=0.01,
                        help='The step rate of gradient ascent')
    parser.add_argument('--img_path', type=str, default='tmp/dogs-vs-cats/train/cats/cat.0.jpg',
                        help='The image path to dream on')
    parser.add_argument('--repair_lost_detail', type=bool, default=True,
                        help='Whether to repair the lost upscaling detail or not')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
