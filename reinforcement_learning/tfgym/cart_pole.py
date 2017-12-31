import argparse
import random
import sys

import numpy as np
import gym
import tensorflow as tf

ACTION_LEFT = 0
ACTION_RIGHT = 1

TERMINATION_REWARD = 200
TERMINATION_ANGLE = 12.0
TERMINATION_POSITION = 2.4


class RandomPolicy(object):
    def predict(self, obs: np.ndarray):
        return random.randint(0, 1)


class Naive1Policy(object):
    def predict(self, obs: np.ndarray):
        angle = obs[2]
        return ACTION_LEFT if angle < 0 else ACTION_RIGHT


class Naive2Policy(object):
    def predict(self, obs: np.ndarray):
        pole_angle = obs[2]
        pole_velo = obs[3]

        # basic action due to angle
        action = ACTION_LEFT if pole_angle < 0 else ACTION_RIGHT

        flip_action = False
        flip_speed_limit = 0.00125
        if abs(pole_angle) < 3.0:
            if pole_angle < 0 and pole_velo > flip_speed_limit:
                flip_action = True
            elif pole_angle > 0 and pole_velo < flip_speed_limit:
                flip_action = True

        return 1 - action if flip_action else action


class Naive3Policy(object):
    def predict(self, obs: np.ndarray):
        cart_pos = float(obs[0])
        pole_angle = obs[2]
        pole_velo = obs[3]

        # basic action due to angle
        action = ACTION_LEFT if pole_angle < 0 else ACTION_RIGHT

        flip_action = False
        flip_speed_limit = 0.00125
        gap = [1.0, 1.165]

        if -gap[1] < cart_pos < -gap[0] or gap[0] < cart_pos < gap[1]:
            return action

        if abs(pole_angle) < 3.0:
            if pole_angle < 0 and pole_velo > flip_speed_limit:
                flip_action = True
            elif pole_angle > 0 and pole_velo < flip_speed_limit:
                flip_action = True

        return 1 - action if flip_action else action


class PolicyGradientsNet(object):
    def __init__(self, n_inputs, n_hidden):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = 1

        self.x_ph = None
        self.action = None
        self.y = None
        self.logits = None
        self.gradients = None
        self.gradient_phs = None
        self.train_op = None
        self.sess = tf.InteractiveSession()
        self.saver = None
        pass

    def forward(self):
        initializer = tf.contrib.layers.variance_scaling_initializer()

        self.x_ph = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
        hidden1 = tf.layers.dense(self.x_ph, self.n_hidden, activation=tf.nn.elu,
                                  kernel_initializer=initializer)
        hidden2 = tf.layers.dense(hidden1, self.n_hidden, activation=tf.nn.elu,
                                  kernel_initializer=initializer)
        self.logits = tf.layers.dense(hidden2, self.n_outputs,
                                      kernel_initializer=initializer)
        outputs = tf.nn.sigmoid(self.logits)

        # select random action based on estimated probabilities
        p_left_right = tf.concat([outputs, 1 - outputs], axis=1)
        self.action = tf.multinomial(tf.log(p_left_right), num_samples=1)

        # act as though the chosen action is the best possible action
        self.y = 1.0 - tf.to_float(self.action)

    def build(self, learning_rate):
        self.forward()

        cost = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,
                                                       logits=self.logits)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cost)
        self.gradients = [grad for grad, var in grads_and_vars]
        self.gradient_phs = []
        grads_and_vars_feed = []
        for grad, var in grads_and_vars:
            gradient_ph = tf.placeholder(tf.float32, shape=grad.get_shape())
            self.gradient_phs.append(gradient_ph)
            grads_and_vars_feed.append((gradient_ph, var))
        self.train_op = optimizer.apply_gradients(grads_and_vars_feed)

    def train(self, env: gym.Env, n_iterations, discount_rate, n_episodes_per_update, n_max_steps,
              save_iterations=50):
        for iteration in range(n_iterations):
            print(iteration)
            all_rewards = []
            all_gradients = []
            for game in range(n_episodes_per_update):
                current_rewards = []
                current_gradients = []

                obs = env.reset()
                for step in range(n_max_steps):
                    action_val, gradients_val = self.sess.run(
                        [self.action, self.gradients],
                        feed_dict={self.x_ph: obs.reshape(1, self.n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)

            # at this point, we have run the policy 10 episodes, and we are ready for a policy update
            all_rewards = PolicyGradientsNet._discount_and_normalize_rewards(all_rewards, discount_rate)
            feed_dict = {}
            for var_index, grad_ph in enumerate(self.gradient_phs):
                # multiply gradients by the action scores, and compute the mean
                mean_gradients = np.mean(
                    [reward * all_gradients[game_index][step][var_index]
                     for game_index, rewards in enumerate(all_rewards)
                     for step, reward in enumerate(rewards)],
                    axis=0)
                feed_dict[grad_ph] = mean_gradients
            self.sess.run(self.train_op, feed_dict)
            if iteration % save_iterations == 0:
                self.saver.save(self.sess, './tmp/cartpole_agent.ckpt')

    def restore(self):
        self.saver = tf.train.Saver()
        latest_cp_path = tf.train.latest_checkpoint('./tmp/')
        if latest_cp_path is None:
            self.sess.run(tf.global_variables_initializer())
            print('Could not find latest checkpoint. Use initializer instead.')
        else:
            self.saver.restore(self.sess, latest_cp_path)
            print('Restored checkpoint: ' + latest_cp_path)

    @staticmethod
    def _discount_rewards(rewards, discount_rate):
        discounted_rewards = np.empty(len(rewards))
        cumultative_rewards = 0
        for step in reversed(range(len(rewards))):
            cumultative_rewards = rewards[step] + cumultative_rewards * discount_rate
            discounted_rewards[step] = cumultative_rewards
        return discounted_rewards

    @staticmethod
    def _discount_and_normalize_rewards(all_rewards, discount_rate):
        all_discounted_rewards = [PolicyGradientsNet._discount_rewards(rewards, discount_rate)
                                  for rewards in all_rewards]
        flat_rewards = np.concatenate(all_discounted_rewards)
        rewards_mean = flat_rewards.mean()
        rewards_std = flat_rewards.std()
        return [(discounted_rewards - rewards_mean) / rewards_std
                for discounted_rewards in all_discounted_rewards]

    def predict(self, obs):
        action_val = self.sess.run(
            self.action,
            feed_dict={self.x_ph: obs.reshape(1, self.n_inputs)})
        return action_val[0][0]


def main(_):
    # https://github.com/openai/gym/wiki/CartPole-v0
    env = gym.make('CartPole-v0')

    if FLAGS.policy == 'naive1':
        policy = Naive1Policy()
    elif FLAGS.policy == 'naive2':
        policy = Naive2Policy()
    elif FLAGS.policy == 'naive3':
        policy = Naive3Policy()
    elif FLAGS.policy == 'pg':
        policy = PolicyGradientsNet(env.observation_space.shape[0], n_hidden=8)
        policy.build(learning_rate=0.01)
        policy.restore()
        policy.train(env, n_iterations=100, discount_rate=0.95, n_episodes_per_update=10, n_max_steps=FLAGS.max_steps)
    else:
        policy = RandomPolicy()

    totals = []
    for episode in range(FLAGS.episodes):
        episode_reward = 0
        obs = env.reset()

        for step in range(FLAGS.max_steps):
            action = policy.predict(obs)

            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if FLAGS.render:
                env.render()

            if done:
                break

        totals.append(episode_reward)

    totals = np.asarray(totals)
    print('Min: {}'.format(np.min(totals)))
    print('Max: {}'.format(np.max(totals)))
    print('Mean: {}'.format(np.mean(totals)))
    print('Std: {}'.format(np.std(totals)))
    print('Successful: {}/{}'.format(len(totals[totals == TERMINATION_REWARD]), FLAGS.episodes))


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
