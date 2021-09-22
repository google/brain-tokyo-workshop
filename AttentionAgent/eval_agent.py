import argparse
import gin
import os
import util
import numpy as np
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-dir', help='Directory of logs.')
    parser.add_argument(
        '--model-filename', default='model.npz',
        help='File name of the model to evaluate.')
    parser.add_argument(
        '--n-episodes', help='Number of episodes to evaluate.',
        type=int, default=3)
    parser.add_argument(
        '--seed', help='Random seed for evaluation.', type=int, default=1)
    config, _ = parser.parse_known_args()
    return config


def main(config):
    logger = util.create_logger(name='test_solution', log_dir=config.log_dir)
    task = util.create_task(logger=logger)
    task.seed(config.seed)

    solution = util.create_solution(device='cpu:0')
    model_file = os.path.join(config.log_dir, config.model_filename)
    solution.load(model_file)

    rewards = []
    time_costs = []
    for ep in range(config.n_episodes):
        start_time = time.perf_counter()
        reward = task.rollout(solution=solution, evaluation=True)
        time_cost = time.perf_counter() - start_time
        rewards.append(reward)
        time_costs.append(time_cost)
        logger.info('Episode: {0}, reward: {1:.2f}'.format(ep + 1, reward))

    logger.info('Avg reward: {0:.2f}, sd: {1:.2f}'.format(
        np.mean(rewards), np.std(rewards)))
    logger.info('Time per rollout: {}s'.format(np.mean(time_costs)))


if __name__ == '__main__':
    args = parse_args()
    gin.parse_config_file(os.path.join(args.log_dir, 'config.gin'))
    main(args)
