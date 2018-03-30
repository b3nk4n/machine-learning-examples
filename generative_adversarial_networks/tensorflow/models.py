from datetime import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import tensorflow as tf

import generative_adversarial_networks.tensorflow.utils as utils


class ConvLayer(object):
    def __init__(self, name, in_depth, out_depth, apply_batch_norm,
                 filter_size=5, stride=2, activation=lambda x: x):
        self.W = tf.get_variable(
            'W_{}'.format(name),
            shape=(filter_size, filter_size, in_depth, out_depth),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.b = tf.get_variable(
            'b_{}'.format(name),
            shape=(out_depth,),
            initializer=tf.zeros_initializer(),
        )
        self.name = name
        self.activation = activation
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        conv_out = tf.nn.conv2d(
            X,
            self.W,
            strides=[1, self.stride, self.stride, 1],
            padding='SAME'
        )
        conv_out = tf.nn.bias_add(conv_out, self.b)

        # apply batch normalization
        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope=self.name,
            )
        return self.activation(conv_out)


class FractionallyStridedConvLayer(object):
    def __init__(self, name, in_depth, out_depth, output_shape, apply_batch_norm,
                 filter_size=5, stride=2, activation=lambda x: x):
        self.W = tf.get_variable(
            'W_{}'.format(name),
            shape=(filter_size, filter_size, out_depth, in_depth),
            initializer=tf.contrib.layers.xavier_initializer()
        )
        self.b = tf.get_variable(
            'b_{}'.format(name),
            shape=(out_depth,),
            initializer=tf.zeros_initializer(),
        )
        self.activation = activation
        self.stride = stride
        self.name = name
        self.output_shape = output_shape
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        conv_out = tf.nn.conv2d_transpose(
            value=X,
            filter=self.W,
            output_shape=self.output_shape,
            strides=[1, self.stride, self.stride, 1],
        )
        conv_out = tf.nn.bias_add(conv_out, self.b)

        # apply batch normalization
        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(
                conv_out,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope=self.name,
            )

        return self.activation(conv_out)


class DenseLayer(object):
    def __init__(self, name, num_in, num_out, apply_batch_norm, activation=lambda x: x):
        self.W = tf.get_variable(
            "W_%s" % name,
            shape=(num_in, num_out),
            initializer=tf.random_normal_initializer(stddev=0.02),
            )
        self.b = tf.get_variable(
            "b_%s" % name,
            shape=(num_out,),
            initializer=tf.zeros_initializer(),
            )
        self.activation = activation
        self.name = name
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]

    def forward(self, X, reuse, is_training):
        a = tf.matmul(X, self.W) + self.b

        # apply batch normalization
        if self.apply_batch_norm:
            a = tf.contrib.layers.batch_norm(
                a,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope=self.name,
            )
        return self.activation(a)


class DCGAN(object):
    def __init__(self, img_size, img_channels, d_sizes, g_sizes, opt_lr, opt_beta1):
        self.img_size = img_size
        self.img_channels = img_channels
        self.latent_dims = g_sizes['z']

        self.X = tf.placeholder(tf.float32, shape=[None, img_size, img_size, img_channels], name='X')
        self.Z = tf.placeholder(tf.float32, shape=[None, self.latent_dims], name='Z')

        self.sample_images = self.build_generator(self.Z, g_sizes)
        logits = self.build_discriminator(self.X, d_sizes)

        with tf.variable_scope('discriminator') as scope:
            scope.reuse_variables()
            sample_logits = self.discriminator_forward(self.sample_images, reuse=True)

        with tf.variable_scope('generator') as scope:
            scope.reuse_variables()
            self.samples_images_test = self.generator_forward(self.Z, reuse=True, is_training=False)

        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logits, labels=tf.zeros_like(sample_logits))

        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)
        self.g_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logits, labels=tf.ones_like(sample_logits)))

        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast(sample_logits < 0, tf.float32)
        num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        # self.d_accuracy = num_correct / tf.cast(num_predictions, tf.float32)
        self.d_accuracy = num_correct / tf.cast((tf.shape(real_predictions)[0] + tf.shape(fake_predictions)[0]), tf.float32)

        # optimizers
        self.d_params = [t for t in tf.trainable_variables() if t.name.startswith('d')]
        self.g_params = [t for t in tf.trainable_variables() if t.name.startswith('g')]

        self.d_train_op = tf.train.AdamOptimizer(opt_lr, beta1=opt_beta1) \
            .minimize(self.d_cost, var_list=self.d_params)
        self.g_train_op = tf.train.AdamOptimizer(opt_lr, beta1=opt_beta1) \
            .minimize(self.g_cost, var_list=self.g_params)

        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)

    def build_generator(self, Z, g_sizes):
        with tf.variable_scope('generator'):
            dims = [self.img_size]
            dim = self.img_size
            for _, _, stride, _ in reversed(g_sizes['conv_layers']):
                dim = int(np.ceil(float(dim) / stride))
                dims.append(dim)

            dims = list(reversed(dims))
            print('dims: {}'.format(dims))
            self.g_dims = dims

            num_in = self.latent_dims
            self.g_dense_layers = []
            count = 0
            for num_out, apply_batch_norm in g_sizes['dense_layers']:
                name = 'g_dense_layer_{}'.format(count)
                count += 1

                layer = DenseLayer(name, num_in, num_out, apply_batch_norm,
                                   activation=tf.nn.relu)
                self.g_dense_layers.append(layer)
                num_in = num_out

            # final dense layer
            num_out = g_sizes['projection'] * dims[0] * dims[0]
            name = 'g_dense_layer_{}'.format(count)
            layer = DenseLayer(name, num_in, num_out, not g_sizes['bn_after_projection'],
                               activation=tf.nn.relu)
            self.g_dense_layers.append(layer)

            # fractually-strided conf-layers
            num_in = g_sizes['projection']
            self.g_conv_layers = []

            # output may use tanh or sigmoid
            num_relus = len(g_sizes['conv_layers']) - 1
            activations = [tf.nn.relu] * num_relus + [g_sizes['output_activation']]

            for i in range(len(g_sizes['conv_layers'])):
                name = 'fs_conv_layers_{}'.format(i)
                num_out, filter_size, stride, apply_batch_norm = g_sizes['conv_layers'][i]
                activation = activations[i]
                batch_size = tf.shape(Z)[0]
                output_shape = [batch_size, dims[i + 1], dims[i + 1], num_out]
                print('num_in: {}, num_out: {}, output_shape: {}'.format(num_in, num_out, output_shape))
                layer = FractionallyStridedConvLayer(
                    name, num_in, num_out, output_shape, apply_batch_norm, filter_size, stride,
                    activation=activation
                )
                self.g_conv_layers.append(layer)
                num_in = num_out

            self.g_sizes = g_sizes
            return self.generator_forward(Z)

    def generator_forward(self, Z, reuse=None, is_training=True):
        output = Z
        for layer in self.g_dense_layers:
            output = layer.forward(output, reuse, is_training)

        output = tf.reshape(output, [-1, self.g_dims[0], self.g_dims[0], self.g_sizes['projection']])

        if self.g_sizes['bn_after_projection']:
            output = tf.contrib.layers.batch_norm(
                output,
                decay=0.9,
                updates_collections=None,
                epsilon=1e-5,
                scale=True,
                is_training=is_training,
                reuse=reuse,
                scope='bn_after_projection'
            )

        for layer in self.g_conv_layers:
            output = layer.forward(output, reuse, is_training)

        return output

    def build_discriminator(self, X, d_sizes):
        with tf.variable_scope('discriminator'):
            self.d_conv_layers = []
            num_in = self.img_channels
            dim = self.img_size
            count = 0
            for num_out, filter_size, stride, apply_batch_norm, in d_sizes['conv_layers']:
                name = 'conv_layer_{}'.format(count)
                count += 1

                layer = ConvLayer(name, num_in, num_out, apply_batch_norm, filter_size, stride,
                                  activation=tf.nn.leaky_relu)
                self.d_conv_layers.append(layer)
                num_in = num_out
                print('dim: {}'.format(dim))
                dim = int(np.ceil(float(dim) / stride))

            num_in = num_in * dim * dim
            self.d_dense_layers = []
            for num_out, apply_batch_norm, in d_sizes['dense_layers']:
                name = 'dense_layer_{}'.format(count)
                count += 1

                layer = DenseLayer(name, num_in, num_out, apply_batch_norm, tf.nn.leaky_relu)
                self.d_dense_layers.append(layer)
                num_in = num_out

            # final logistic layer
            name = 'dense_layer_{}'.format(count)
            self.d_final_layer = DenseLayer(name, num_in, 1, False)

            logits = self.discriminator_forward(X)
            return logits

    def discriminator_forward(self, X, reuse=None, is_training=True):
        output = X
        for layer in self.d_conv_layers:
            output = layer.forward(output, reuse, is_training)
        output = tf.contrib.layers.flatten(output)
        for layer in self.d_dense_layers:
            output = layer.forward(output, reuse, is_training)
        logits = self.d_final_layer.forward(output, reuse, is_training)
        return logits

    def fit(self, X, epochs, batch_size, save_sample_interval=100, output_root='tmp'):
        d_costs = []
        g_costs = []

        n = len(X)
        n_batches = n // batch_size
        step = 0
        for i in range(epochs):
            print('Starting epoche: {}'.format(i))
            np.random.shuffle(X)
            for j in range(n_batches):
                t0 = datetime.now()

                batch = X[j * batch_size:(j + 1) * batch_size]
                if type(X[0]) is str:
                    # celeb
                    batch = utils.files2images(batch)

                Z = np.random.uniform(-1, 1, size=(batch_size, self.latent_dims))

                # discriminator training
                _, d_cost, d_acc = self.sess.run([self.d_train_op, self.d_cost, self.d_accuracy], {
                    self.X: batch, self.Z: Z
                })
                d_costs.append(d_cost)

                # generator training
                _, g_cost1 = self.sess.run([self.g_train_op, self.g_cost], {
                    self.Z: Z
                })
                _, g_cost2 = self.sess.run([self.g_train_op, self.g_cost], {
                    self.Z: Z
                })
                g_costs.append((g_cost1 + g_cost2) / 2)

                print('Batch {}/{}: dt: {}, d_acc: {:.2f}'.format(j + 1, n_batches, datetime.now() - t0, d_acc))

                step += 1
                if step % save_sample_interval == 0:
                    print('Saving a sample...')
                    n_samples = 64
                    samples = self.sample(n_samples)
                    self._save_samples_image(os.path.join(output_root, 'samples_{:05d}.png'.format(step)), samples)

        plt.clf()
        plt.plot(d_costs, label='Discriminator Cost')
        plt.plot(g_costs, label='Generator Cost')
        plt.legend()
        plt.savefig(os.path.join(output_root, 'training_costs.png'))

    def _save_samples_image(self, filepath, samples):
        n_samples = samples.shape[0]
        n_samples_sqrt = int(np.sqrt(n_samples))
        d = samples.shape[1]
        if samples.shape[-1] == 1:
            # gray image: (N x N)
            samples = samples.reshape(n_samples, d, d)
            flat_image = np.empty([n_samples_sqrt * d, n_samples_sqrt * d])

            k = 0
            for i in range(n_samples_sqrt):
                for j in range(n_samples_sqrt):
                    flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k].reshape(d, d)
                    k += 1
        elif samples.shape[-1] == 3:
            # color image: (N x N x 3)
            flat_image = np.empty([n_samples_sqrt * d, n_samples_sqrt * d, 3])

            k = 0
            for i in range(n_samples_sqrt):
                for j in range(n_samples_sqrt):
                    flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k]
                    k += 1
        else:
            raise Exception('Invalid image shape!')

        sp.misc.imsave(filepath, flat_image)

    def sample(self, n):
        Z = np.random.uniform(-1, 1, size=[n, self.latent_dims])
        samples = self.sess.run(self.samples_images_test, {
            self.Z: Z
        })
        return samples
