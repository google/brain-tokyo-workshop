import argparse
import gin
import os
import misc.utility
import numpy as np
import shutil


def main(config):
    """Test policy."""

    logger = misc.utility.create_logger(name='test_solution', debug=True)
    task = misc.utility.create_task(logger=logger)
    task.seed(config.seed)

    solution = misc.utility.create_solution()
    model_file = os.path.join(config.log_dir, config.model_filename)
    solution.load(model_file)

    rewards = []
    for ep in range(config.n_episodes):
        if config.save_screens and hasattr(solution, 'set_log_dir'):
            dd = os.path.join(config.log_dir, 'test_ep{}'.format(ep + 1))
            if os.path.exists(dd):
                shutil.rmtree(dd)
            solution.set_log_dir(dd)
        if config.save_screens and hasattr(task, 'set_video_dir'):
            task.set_video_dir(
                os.path.join(config.log_dir, 'test_ep{}/video'.format(ep + 1))
            )

        reward = task.roll_out(solution=solution, evaluate=True)
        logger.info('Episode: {0}, reward: {1:.2f}'.format(ep + 1, reward))
        rewards.append(reward)

    logger.info('Avg reward: {0:.2f}, sd: {1:.2f}'.format(
        np.mean(rewards), np.std(rewards)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log-dir', help='Directory of logs.')
    parser.add_argument(
        '--model-filename', default='model.npz',
        help='File name of the model to evaluate.')
    parser.add_argument(
        '--render', help='Whether to render while evaluation.', default=False,
        action='store_true')
    parser.add_argument(
        '--save-screens', help='Whether to save screenshots.', default=False,
        action='store_true')
    parser.add_argument(
        '--overplot', help='Whether to render overplotted image.',
        default=False, action='store_true')
    parser.add_argument(
        '--n-episodes', help='Number of episodes to evaluate.',
        type=int, default=3)
    parser.add_argument(
        '--seed', help='Random seed for evaluation.', type=int, default=1)
    args, _ = parser.parse_known_args()

    gin.parse_config_file(os.path.join(args.log_dir, 'config.gin'))
    gin.bind_parameter("utility.create_task.render", args.render)
    if args.overplot:
        gin.bind_parameter(
            "torch_solutions.VisionTaskSolution.show_overplot", True)

    main(args)
