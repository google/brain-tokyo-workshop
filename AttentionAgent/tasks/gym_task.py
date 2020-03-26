import cv2
import gin
import gym
from gym import spaces
import numpy as np
import os
import tasks.abc_task
import time


class GymTask(tasks.abc_task.BaseTask):
    """OpenAI gym tasks."""

    def __init__(self):
        self._env = None
        self._render = False
        self._logger = None

    def create_task(self, **kwargs):
        raise NotImplementedError()

    def seed(self, seed):
        if isinstance(self, TakeCoverTask):
            self._env.game.set_seed(seed)
        else:
            self._env.seed(seed)

    def reset(self):
        return self._env.reset()

    def step(self, action, evaluate):
        return self._env.step(action)

    def close(self):
        self._env.close()

    def _process_reward(self, reward, done, evaluate):
        return reward

    def _process_action(self, action):
        return action

    def _process_observation(self, observation):
        return observation

    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        return done

    def _show_gui(self):
        if hasattr(self._env, 'render'):
            self._env.render()

    def roll_out(self, solution, evaluate):
        ob = self.reset()
        ob = self._process_observation(ob)
        if hasattr(solution, 'reset'):
            solution.reset()

        start_time = time.time()

        rewards = []
        done = False
        step_cnt = 0
        while not done:
            action = solution.get_output(inputs=ob, update_filter=not evaluate)
            action = self._process_action(action)
            ob, r, done, _ = self.step(action, evaluate)
            ob = self._process_observation(ob)

            if self._render:
                self._show_gui()

            step_cnt += 1
            done = self._overwrite_terminate_flag(r, done, step_cnt, evaluate)
            step_reward = self._process_reward(r, done, evaluate)
            rewards.append(step_reward)

        time_cost = time.time() - start_time
        actual_reward = np.sum(rewards)
        if hasattr(self, '_logger') and self._logger is not None:
            self._logger.info(
                'Roll-out time={0:.2f}s, steps={1}, reward={2:.2f}'.format(
                    time_cost, step_cnt, actual_reward))

        return actual_reward


@gin.configurable
class TakeCoverTask(GymTask):
    """VizDoom take cover task."""

    def create_task(self, **kwargs):
        from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
        if 'render' in kwargs:
            self._render = kwargs['render']
        if 'logger' in kwargs:
            self._logger = kwargs['logger']
        self._env = DoomTakeCoverEnv()
        return self

    def _process_action(self, action):
        # Follow the code in world models.
        action_to_apply = [0] * 43
        threshold = 0.3333
        if action > threshold:
            action_to_apply[10] = 1
        if action < -threshold:
            action_to_apply[11] = 1
        return action_to_apply

    def set_video_dir(self, video_dir):
        from gym.wrappers import Monitor
        self._env = Monitor(
            env=self._env,
            directory=video_dir,
            video_callable=lambda x: True
        )


@gin.configurable
class CarRacingTask(GymTask):
    """Gym CarRacing-v0 task."""

    def __init__(self):
        super(CarRacingTask, self).__init__()
        self._max_steps = 0
        self._neg_reward_cnt = 0
        self._neg_reward_cap = 0
        self._action_high = np.array([1., 1., 1.])
        self._action_low = np.array([-1., 0., 0.])

    def _process_action(self, action):
        return (action * (self._action_high - self._action_low) / 2. +
                (self._action_high + self._action_low) / 2.)

    def reset(self):
        ob = super(CarRacingTask, self).reset()
        self._neg_reward_cnt = 0
        return ob

    def _overwrite_terminate_flag(self, reward, done, step_cnt, evaluate):
        if evaluate:
            return done
        if reward < 0:
            self._neg_reward_cnt += 1
        else:
            self._neg_reward_cnt = 0
        too_many_out_of_tracks = 0 < self._neg_reward_cap < self._neg_reward_cnt
        too_many_steps = 0 < self._max_steps <= step_cnt
        return done or too_many_out_of_tracks or too_many_steps

    def create_task(self, **kwargs):
        self._env = gym.make('CarRacing-v0')
        if 'render' in kwargs:
            self._render = kwargs['render']
        if 'out_of_track_cap' in kwargs:
            self._neg_reward_cap = kwargs['out_of_track_cap']
        if 'max_steps' in kwargs:
            self._max_steps = kwargs['max_steps']
        if 'logger' in kwargs:
            self._logger = kwargs['logger']
        return self

    def set_video_dir(self, video_dir):
        from gym.wrappers import Monitor
        self._env = Monitor(
            env=self._env,
            directory=video_dir,
            video_callable=lambda x: True
        )

