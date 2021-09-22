import gin
import gym
import numpy as np
import time
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
class CartPoleSwingUpTask(RLTask):
    """Car-pole swing up task."""

    def __init__(self, shuffle_on_reset=False, render=False, v=True):
        super(CartPoleSwingUpTask, self).__init__(v=v)
        self.shuffle_on_reset = shuffle_on_reset
        self.perm_ix = 0
        self.render = render
        self.env = CartPoleSwingUpHarderEnv()
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.render:
            self.env.render('human')

    def reset_for_rollout(self):
        self.perm_ix = np.arange(self.env.observation_space.shape[0])
        if self.shuffle_on_reset:
            np.random.shuffle(self.perm_ix)
        if self.verbose:
            print('perm_ix: {}'.format(self.perm_ix))
        return super(CartPoleSwingUpTask, self).reset_for_rollout()

    def modify_obs(self, obs):
        return obs[self.perm_ix]


@gin.configurable
class CarRacingTask(RLTask):
    """Gym CarRacing-v0 task."""

    def __init__(self, out_of_track_cap=20):
        super(CarRacingTask, self).__init__()
        self._max_steps = 1000
        self._neg_reward_cnt = 0
        self._neg_reward_cap = out_of_track_cap
        self._action_high = np.array([1., 1., 1.])
        self._action_low = np.array([-1., 0., 0.])
        self.env = gym.make('CarRacing-v0')

    def modify_action(self, act):
        return (act * (self._action_high - self._action_low) / 2. +
                (self._action_high + self._action_low) / 2.)

    def reset_for_rollout(self):
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
