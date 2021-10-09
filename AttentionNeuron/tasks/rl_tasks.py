from collections import deque
import cv2
import os
import gin
import gym
import numpy as np
import time
import pybullet_envs
from tasks import atari_wrappers
from tasks.base_task import BaseTask
from tasks.cartpole_env import CartPoleSwingUpHarderEnv


class RLTask(BaseTask):
    """RL base task."""

    def __init__(self, v=True):
        self.env = None
        self.render = False
        self.step_cnt = 0
        self.eval_mode = False
        self.verbose = v

    def reset_for_rollout(self):
        self.step_cnt = 0

    def seed(self, seed=None):
        return self.env.seed(seed)

    def modify_obs(self, obs):
        return obs

    def modify_action(self, act):
        return act

    def modify_reward(self, reward, done):
        return reward

    def modify_done(self, reward, done):
        return done

    def show_gui(self):
        if self.render and hasattr(self.env, 'render'):
            return self.env.render()

    def close(self):
        self.env.close()

    def rollout(self, solution, evaluation=False):
        self.eval_mode = evaluation
        self.reset_for_rollout()
        solution.reset()
        if hasattr(self, 'register_solution'):
            self.register_solution(solution)

        start_time = time.time()

        obs = self.env.reset()
        obs = self.modify_obs(obs)
        self.show_gui()
        ep_reward = 0
        done = False
        while not done:
            action = solution.get_action(obs)
            action = self.modify_action(action)
            obs, reward, done, info = self.env.step(action)
            obs = self.modify_obs(obs)
            reward = self.modify_reward(reward, done)
            done = self.modify_done(reward, done)
            self.step_cnt += 1
            ep_reward += reward
            self.show_gui()

        time_cost = time.time() - start_time
        if self.verbose:
            print('Rollout time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
                time_cost, self.step_cnt, ep_reward))

        return ep_reward


@gin.configurable
class PyBulletTask(RLTask):

    def __init__(self, env_name, shuffle_on_reset=False, render=False, v=True):
        super(PyBulletTask, self).__init__(v=v)
        self.env_name = env_name
        self.shuffle_on_reset = shuffle_on_reset
        self.perm_ix = 0
        self.render = render
        self.env = gym.make(self.env_name)
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.render:
            self.env.render('human')

    def reset_for_rollout(self):
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.shuffle_on_reset:
            np.random.shuffle(self.perm_ix)
        if self.verbose:
            print('perm_ix: {}'.format(self.perm_ix))
        return super(PyBulletTask, self).reset_for_rollout()

    def modify_reward(self, reward, done):
        if self.eval_mode:
            return reward
        else:
            return max(0, sum(self.env.rewards[1:]))

    def modify_obs(self, obs):
        return obs[self.perm_ix]

    def show_gui(self):
        if self.render:
            time.sleep(0.01)
            return super(PyBulletTask, self).show_gui()


@gin.configurable
class CartPoleSwingUpTask(RLTask):
    """Car-pole swing up task."""

    def __init__(self,
                 shuffle_on_reset=False,
                 render=False,
                 v=True,
                 num_noise_channels=0):
        super(CartPoleSwingUpTask, self).__init__(v=v)
        self.shuffle_on_reset = shuffle_on_reset
        self.perm_ix = 0
        self.render = render
        self.env = CartPoleSwingUpHarderEnv()
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        self.noise_std = 0.1
        self.num_noise_channels = num_noise_channels
        self.rnd = np.random.RandomState(seed=0)

    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed=seed)
        return super(CartPoleSwingUpTask, self).seed(seed)

    def reset_for_rollout(self):
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.shuffle_on_reset:
            self.rnd.shuffle(self.perm_ix)
        if self.verbose:
            print('perm_ix: {}'.format(self.perm_ix))
        return super(CartPoleSwingUpTask, self).reset_for_rollout()

    def modify_obs(self, obs):
        obs = obs[self.perm_ix]
        if self.num_noise_channels > 0:
            noise_obs = self.rnd.randn(self.num_noise_channels) * self.noise_std
            obs = np.concatenate([obs, noise_obs], axis=0)
        return obs


@gin.configurable
class CarRacingTask(RLTask):
    """Gym CarRacing-v0 task."""

    def __init__(self,
                 bkg=None,
                 permute_obs=False,
                 patch_size=6,
                 out_of_track_cap=20,
                 stack_k_frames=0,
                 render=False):
        super(CarRacingTask, self).__init__()

        self.permute_obs = permute_obs
        self.patch_size = patch_size
        self.bkg = bkg
        bkg_file = os.path.join(
            os.path.dirname(__file__), 'bkg/{}.jpg'.format(self.bkg))
        if os.path.exists(bkg_file):
            self.bkg = cv2.resize(cv2.imread(bkg_file), (96, 96))[:, :, ::-1]
        else:
            self.bkg = None
        self.original_obs = None
        self.shuffled_obs = None
        self.obs_perm_ix = np.arange((96 // self.patch_size)**2)
        self.rnd = np.random.RandomState(seed=0)
        self.solution = None

        self.render = render
        self._max_steps = 1000
        self._neg_reward_cnt = 0
        self._neg_reward_cap = out_of_track_cap
        self._action_high = np.array([1., 1., 1.])
        self._action_low = np.array([-1., 0., 0.])
        self.env = gym.make('CarRacing-v0')
        self.stack_k_frames = stack_k_frames
        if self.stack_k_frames > 0:
            self.obs_stack = deque(maxlen=self.stack_k_frames)
            
    def seed(self, seed=None):
        self.rnd = np.random.RandomState(seed=seed)
        return super(CarRacingTask, self).seed(seed)

    def modify_action(self, act):
        return (act * (self._action_high - self._action_low) / 2. +
                (self._action_high + self._action_low) / 2.)

    def reset_for_rollout(self):
        self.original_obs = None
        self.shuffled_obs = None
        self.obs_perm_ix = np.arange((96 // self.patch_size)**2)
        if self.permute_obs:
            self.rnd.shuffle(self.obs_perm_ix)
        if self.stack_k_frames > 0:
            self.obs_stack = deque(maxlen=self.stack_k_frames)
        self._neg_reward_cnt = 0
        return super(CarRacingTask, self).reset_for_rollout()

    def modify_done(self, reward, done):
        if self.eval_mode:
            return done
        if reward < 0:
            self._neg_reward_cnt += 1
        else:
            self._neg_reward_cnt = 0
        too_many_out_of_tracks = 0 < self._neg_reward_cap < self._neg_reward_cnt
        too_many_steps = 0 < self._max_steps <= self.step_cnt
        return done or too_many_out_of_tracks or too_many_steps

    def shuffle_obs_patches(self, obs):
        shuffled_obs = np.zeros_like(obs)
        p_size = self.patch_size
        num_patches_per_dim = 96 // p_size
        for pstart_r in range(num_patches_per_dim):
            for pstart_c in range(num_patches_per_dim):
                ix = pstart_r * num_patches_per_dim + pstart_c
                shuffled_ix = self.obs_perm_ix[ix]
                spstart_r = shuffled_ix // num_patches_per_dim
                spstart_c = shuffled_ix % num_patches_per_dim
                shuffled_obs[
                    pstart_r * p_size:(pstart_r + 1) * p_size,
                    pstart_c * p_size:(pstart_c + 1) * p_size
                ] = obs[
                    spstart_r * p_size:(spstart_r + 1) * p_size,
                    spstart_c * p_size:(spstart_c + 1) * p_size
                ]
        return shuffled_obs

    def modify_obs(self, obs):
        if self.bkg is not None:
            mask = ((obs[:, :, 0] == 102) &
                    (obs[:, :, 1] == 204) &
                    (obs[:, :, 2] == 102))
            mask |= ((obs[:, :, 0] == 102) &
                     (obs[:, :, 1] == 230) &
                     (obs[:, :, 2] == 102))
            obs[:, :, 0][mask] = self.bkg[:, :, 0][mask]
            obs[:, :, 1][mask] = self.bkg[:, :, 1][mask]
            obs[:, :, 2][mask] = self.bkg[:, :, 2][mask]

        # Keep original and shuffled screens for visualization.
        self.original_obs = obs
        self.shuffled_obs = obs

        if self.permute_obs:
            self.shuffled_obs = self.shuffle_obs_patches(obs)

        if self.stack_k_frames > 0:
            gray_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            gray_obs[-12:] = 0  # Zero-out the indicator.
            if self.permute_obs:
                gray_obs = self.shuffle_obs_patches(gray_obs)
            while len(self.obs_stack) < self.stack_k_frames:
                self.obs_stack.append(gray_obs)
            self.obs_stack.append(gray_obs)
            obs = np.stack(self.obs_stack)
            return obs
        else:
            return self.shuffled_obs

    def register_solution(self, solution):
        self.solution = solution

    def plot_white_patches(self, img, white_patch_ix):
        white_patch = np.ones([self.patch_size, self.patch_size, 3]) * 255
        num_patches = 96 // self.patch_size
        for ix in white_patch_ix:
            row_ix = ix // num_patches
            col_ix = ix % num_patches
            row_ss = row_ix * self.patch_size
            col_ss = col_ix * self.patch_size
            row_ee = row_ss + self.patch_size
            col_ee = col_ss + self.patch_size
            img[row_ss:row_ee, col_ss:col_ee] = (
                    0.5 * img[row_ss:row_ee, col_ss:col_ee] + 0.5 * white_patch)
        return img.astype(np.uint8)

    def show_gui(self):
        if self.render:
            if hasattr(self.solution, 'attended_patch_ix'):
                attended_patch_ix = self.solution.attended_patch_ix
            else:
                attended_patch_ix = None

            obs = self.shuffled_obs.copy()
            if attended_patch_ix is not None:
                obs = self.plot_white_patches(
                    img=obs, white_patch_ix=attended_patch_ix)

            org_obs = self.original_obs.copy()
            if attended_patch_ix is not None:
                org_obs = self.plot_white_patches(
                    img=org_obs,
                    white_patch_ix=[self.obs_perm_ix[i]
                                    for i in attended_patch_ix])

            img = np.concatenate([org_obs, obs], axis=1)
            img = cv2.resize(img, (800, 400))[:, :, ::-1]
            cv2.imshow('render', img)
            cv2.waitKey(1)
        return super(CarRacingTask, self).show_gui()


@gin.configurable
class PuzzlePongTask(RLTask):
    """Atari Pong."""

    def __init__(self,
                 permute_obs=False,
                 patch_size=6,
                 occlusion_ratio=0.,
                 render=False):
        super(PuzzlePongTask, self).__init__()
        self.render = render
        self.occlusion_ratio = occlusion_ratio
        self.env = atari_wrappers.wrap_deepmind(
            env=atari_wrappers.make_atari(env_id='PongNoFrameskip-v4'),
            episode_life=False,
            clip_rewards=False,
            flicker=False,
            frame_stack=True,
            permute_obs=permute_obs,
            patch_size=patch_size,
            rand_zero_out_ratio=occlusion_ratio,
        )

    def modify_obs(self, obs):
        # Convert from LazyFrames to numpy array.
        obs = np.array(obs)
        # Uncomment to confirm the env is indeed passing shuffled obs.
        # cv2.imshow('Pong debug', cv2.resize(obs[0], (200, 200)))
        # cv2.waitKey(1)
        if 0. < self.occlusion_ratio < 1.:
            return {'obs': obs, 'patches_to_use': self.env.patch_to_keep_ix}
        else:
            return obs
