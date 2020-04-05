import argparse
import car_racing_variants
import gym
import numpy as np


def main(config):
    """Test customized CarRacing envs."""

    env = gym.make('CarRacing{}-v0'.format(config.env_string))
    env.seed(config.seed)

    while True:
        ob = env.reset()
        done = False
        step = 0
        while not done and 0 <= step <= config.step_limit:
          ob, reward, done, _ = env.step(env.action_space.sample())
          step += 1
          env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env-string', default='Color',
        help='Environment string. E.g. Color, Color3, Bar, Blob, Noise.')
    parser.add_argument(
        '--seed', default=1, type=int,
        help='Random seed for the environment.')
    parser.add_argument(
        '--step-limit', default=100, type=int,
        help='Step limit. A negative value disables this limit.')
    args, _ = parser.parse_known_args()
    main(args)

