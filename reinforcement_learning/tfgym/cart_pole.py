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


def random_policy() -> int:
    return random.randint(0, 1)


def naive1_policy(obs: np.ndarray) -> int:
    angle = obs[2]
    return ACTION_LEFT if angle < 0 else ACTION_RIGHT


def naive2_policy(obs: np.ndarray) -> int:
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


def naive3_policy(obs: np.ndarray) -> int:
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


def main(_):
    # https://github.com/openai/gym/wiki/CartPole-v0
    env = gym.make('CartPole-v0')
    totals = []
    for episode in range(FLAGS.episodes):
        episode_reward = 0
        obs = env.reset()

        for step in range(FLAGS.max_steps):
            if FLAGS.policy == 'naive1':
                action = naive1_policy(obs)
            elif FLAGS.policy == 'naive2':
                action = naive2_policy(obs)
            elif FLAGS.policy == 'naive3':
                action = naive3_policy(obs)
            else:
                action = random_policy()

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
    parser.add_argument('--episodes', type=int, default=500,
                        help='The number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='The max number of steps per episode')
    parser.add_argument('--policy', type=str, default='naive3',
                        help='The policy to use')
    parser.add_argument('--render', type=bool, default=False,
                        help='Set True to render the scene')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
