import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.keras import applications
from tensorflow.contrib.keras import backend as K
from tensorflow.contrib.keras import preprocessing

import cnn_classification.keras.dogs_cats_dataset as dataset
import cnn_classification.keras.utils as utils


def show_top_predictions(preds, top=3):
    decoded_preds = applications.vgg16.decode_predictions(preds, top=top)[0]
    for i, (_, clazz, prob) in enumerate(decoded_preds):
        print('{}. {:20s}: {:.4f}'.format(i + 1, clazz, prob))


def get_top_prediction(preds):
    decoded_preds = applications.vgg16.decode_predictions(preds, top=1)[0]
    clazz = decoded_preds[0][1]
    prob = decoded_preds[0][2]
    return clazz, prob


def normalize_heatmap(heatmap):
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return heatmap


def superimpose_image_with_heatmap(dest_img_path, src_img_path, heatmap, strength=0.4):
    img = cv2.imread(src_img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * strength + img

    cv2.imwrite(dest_img_path, superimposed_img)


def gradCAM(model, dog_x, class_idx):
    model_class_output = model.output[:, class_idx]
    print('model_class_output shape', model_class_output.shape)
    last_conv_layer = model.get_layer('block5_conv3')

    grads_list = K.gradients(model_class_output, last_conv_layer.output)
    grads = grads_list[0]
    # grads shape: (?, 14, 14, 512)

    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # pooled_grads shape: (512,)

    iterate = K.function(inputs=[model.input],
                         outputs=[pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([dog_x])
    # conv_layer_output_value shape: (14, 14, 512)

    for i in range(last_conv_layer.filters):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    # heatmap shape: (14, 14)
    heatmap = normalize_heatmap(heatmap)
    return heatmap


def run(i, model, train_dogs_dir, tag):
    img_filename = utils.listdir(train_dogs_dir, recursive=True)[i]
    img_path = os.path.join(train_dogs_dir, img_filename)
    img_dog = preprocessing.image.load_img(img_path, target_size=(224, 224))

    os.makedirs('tmp/output', exist_ok=True)
    processed_img_path = 'tmp/output/processed-{}-{}.png'.format(tag, i)
    img_dog.save(processed_img_path)

    dog_x = preprocessing.image.img_to_array(img_dog)
    dog_x = np.expand_dims(dog_x, axis=0)
    dog_x = applications.vgg16.preprocess_input(dog_x)

    preds = model.predict(dog_x)
    show_top_predictions(preds, top=3)
    clazz, prob = get_top_prediction(preds)

    class_idx = np.argmax(preds[0])

    heatmap = gradCAM(model, dog_x, class_idx)

    if FLAGS.show_intermediate_heatmap:
        plt.matshow(heatmap)
        plt.show()

    superimposed_output_path = 'tmp/output/superimposed-{}-{}-{}-{}.png'.format(tag, i, clazz, int(round(prob * 100)))
    superimpose_image_with_heatmap(superimposed_output_path, img_path, heatmap, strength=0.4)


def main(_):
    model = applications.VGG16(weights='imagenet')

    # load images for visualization
    train_dir, _, _ = dataset.prepare(train_size=2 * FLAGS.n_per_class, valid_size=0, test_size=0)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    train_cats_dir = os.path.join(train_dir, 'cats')

    for i in range(FLAGS.n_per_class):
        run(i, model, train_dogs_dir, 'dog')
        run(i, model, train_cats_dir, 'cat')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_per_class', type=int, default=10,
                        help='The number of animals per class')
    parser.add_argument('--show_intermediate_heatmap', type=bool, default=False,
                        help='Show intermediate heatmap with matplotlib')
    FLAGS, unparsed = parser.parse_known_args()
    main([sys.argv[0]] + unparsed)
