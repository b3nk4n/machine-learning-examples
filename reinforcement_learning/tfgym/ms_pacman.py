import argparse
import os
import sys

import gym
import numpy as np
import tensorflow as tf

from collections import deque


def preprocess_observation(obs):
    mspacman_color = np.array([210, 164, 74])
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.mean(axis=2)  # greyscale
    img[img == mspacman_color] = 0  # improve contrast
    img.reshape(88, 80, 1)
    img = (img - 128) / 128 - 1  # normalize [-1, 1]
    return img.reshape(88, 80, 1)


class DQN(object):
    def __init__(self, n_outputs, replay_memory_size):
        self.input_height = 88
        self.input_width = 80
        self.input_channels = 1

        self.conv_n_maps = [32, 64, 64]
        self.conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
        self.conv_strides = [4, 2, 1]
        self.conv_paddings = ["SAME"] * 3
        self.conv_activation = [tf.nn.relu] * 3
        self.n_hidden_in = 64 * 11 * 10
        self.n_hidden = 512
        self.hidden_activation = tf.nn.relu
        self.n_outputs = n_outputs  # env.action_space.n
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        self.replay_memory = deque([], maxlen=replay_memory_size)

        self.eps_min = 0.05
        self.eps_max = 1.0
        self.eps_decay_steps = 50000

        self.x_state_ph = None
        self.x_action_ph = None
        self.y_ph = None
        self.global_step = None
        self.train_op = None
        self.copy_critic_to_actor = None
        self.actor_q_values = None

    def forward(self, x_state, name):
        prev_layer = x_state
        conv_layers = []
        with tf.variable_scope(name) as scope:
            for n_maps, kernel_size, stride, padding, activation in zip(
                    self.conv_n_maps, self.conv_kernel_sizes, self.conv_strides, self.conv_paddings, self.conv_activation):
                prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size,
                                              strides=stride, padding=padding, activation=activation,
                                              kernel_initializer=self.initializer)
                conv_layers.append(prev_layer)
            last_conv_layer_flat = tf.layers.flatten(prev_layer)
            hidden = tf.layers.dense(last_conv_layer_flat, self.n_hidden,
                                     activation=self.hidden_activation,
                                     kernel_initializer=self.initializer)
            outputs = tf.layers.dense(hidden, self.n_outputs,
                                      kernel_initializer=self.initializer)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        # strip the scope name
        trainable_vars_by_name = {var.name[len(scope.name):]: var
                                  for var in trainable_vars}
        return outputs, trainable_vars_by_name

    def build(self, learning_rate):
        self.x_state_ph = tf.placeholder(tf.float32, shape=[None, self.input_height, self.input_width, self.input_channels])
        self.actor_q_values, actor_vars = self.forward(self.x_state_ph, 'q_networks/actor')
        critic_q_values, critic_vars = self.forward(self.x_state_ph, 'q_networks/critic')

        copy_ops = [actor_var.assign(critic_vars[var_name])
                    for var_name, actor_var in actor_vars.items()]
        self.copy_critic_to_actor = tf.group(*copy_ops)

        self.x_action_ph = tf.placeholder(tf.int32, shape=[None])
        q_value = tf.reduce_sum(critic_q_values * tf.one_hot(self.x_action_ph, self.n_outputs), axis=1, keep_dims=True)

        self.y_ph = tf.placeholder(tf.float32, shape=[None, 1])
        cost = tf.reduce_mean(tf.square(self.y_ph - q_value))
        self.global_step = tf.train.create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = optimizer.minimize(cost, self.global_step)  # gstep missing for auto increment?

    def _sample_memories(self, batch_size):
        indices = np.random.permutation(len(self.replay_memory))[:batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for idx in indices:
            memory = self.replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def _epsilon_greedy(self, q_values, step):
        epsilon = max(self.eps_min, self.eps_max - (self.eps_max - self.eps_min) * step / self.eps_decay_steps)
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            return np.argmax(q_values)

    def train(self, env: gym.Env):
        saver = tf.train.Saver()

        n_steps = 10000  # total number of training steps
        training_start = 1000  # start training after 1000 game iterations
        training_interval = 3
        save_steps = 50
        copy_steps = 25
        discount_rate = 0.95
        skip_start = 90
        batch_size = 50
        iteration = 0
        checkpoint_path = './tmp/mspacman_agent.ckpt'
        done = True
        state = None

        with tf.Session() as sess:
            if os.path.isfile(checkpoint_path):
                saver.restore(sess, checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            while True:
                print(iteration)
                step = tf.train.global_step(sess, self.global_step)
                if step >= n_steps:
                    break
                iteration += 1
                if done:
                    obs = env.reset()
                    for skip in range(skip_start):
                        obs, reward, done, info = env.step(0)
                    state = preprocess_observation(obs)

                # actor evaluates what to do
                q_values = sess.run(self.actor_q_values, feed_dict={self.x_state_ph: [state]})
                action = self._epsilon_greedy(q_values, step)

                # actor plays
                obs, reward, done, info = env.step(action)
                if FLAGS.render and iteration >= 3000:
                    env.render()
                next_state = preprocess_observation(obs)

                # let's memorize what just happened
                self.replay_memory.append((state, action, reward, next_state, 1.0 - done))
                state = next_state

                if iteration < training_start or iteration % training_interval != 0:
                    continue

                # critic learns
                x_state_val, x_action_val, rewards, x_next_state_val, continues = (
                    self._sample_memories(batch_size))
                next_q_values = sess.run(self.actor_q_values, feed_dict={self.x_state_ph: x_next_state_val})
                max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                y_val = rewards + continues * discount_rate * max_next_q_values
                sess.run(self.train_op, feed_dict={self.x_state_ph: x_state_val,
                                                   self.x_action_ph: x_action_val,
                                                   self.y_ph: y_val})

                if step % copy_steps == 0:
                    sess.run(self.copy_critic_to_actor)

                if step % save_steps == 0:
                    saver.save(sess, checkpoint_path)

    def predict(self, obs):
        # TODO implement
        pass


def main(_):
    env = gym.make('MsPacman-v0')

    dqn = DQN(n_outputs=env.action_space.n,
              replay_memory_size=10000)
    dqn.build(learning_rate=0.001)
    dqn.train(env)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100,
                        help='The number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='The max number of steps per episode')
    parser.add_argument('--policy', type=str, default='pg',
                        help='The policy to use')
    parser.add_argument('--render', type=bool, default=True,
                        help='Set True to render the scene')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
