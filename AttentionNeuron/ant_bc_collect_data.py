import gin
import os
import util
import numpy as np
import gym


def main(log_dir):
    logger = util.create_logger(name='data_collection')

    solution = util.create_solution(device='cpu:0')
    model_file = os.path.join(log_dir, 'model.npz')
    solution.load(model_file)

    trajectories = []
    env = gym.make('AntBulletEnv-v0')

    # Collect trajectories from rollouts.
    max_ep_cnt = 1000
    traj_len = 500
    ep_saved = 0
    while ep_saved < max_ep_cnt:
        ep_reward = 0
        ep_steps = 0
        obs = env.reset()
        prev_act = np.zeros(8)
        ep_traj = []
        done = False
        while not done and ep_steps < traj_len:
            act = solution.get_action(obs)
            ep_traj.append(np.concatenate([prev_act, obs, act], axis=0))
            obs, reward, done, info = env.step(act)
            ep_reward += reward
            ep_steps += 1
        logger.info(
            'Episode:{0}, steps:{1},  reward:{2:.2f}'.format(
                ep_saved + 1, ep_steps, ep_reward)
        )
        if ep_steps >= traj_len:
            trajectories.append(np.vstack(ep_traj))
            ep_saved += 1
        else:
            logger.info('Trajectory too short, discard.')

    trajectories = np.stack(trajectories)
    logger.info('trajectories.shape={}'.format(trajectories.shape))
    np.savez(os.path.join(log_dir, 'data.npz'), data=trajectories)


if __name__ == '__main__':
    model_dir = 'pretrained/ant_mlp'
    gin.parse_config_file(os.path.join(model_dir, 'config.gin'))
    main(model_dir)
