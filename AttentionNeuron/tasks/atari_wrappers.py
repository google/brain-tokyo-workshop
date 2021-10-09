"""This code is based on
https://github.com/pfnet/pfrl/blob/master/pfrl/wrappers/atari_wrappers.py
and
https://github.com/pfnet/pfrl/blob/master/pfrl/wrappers/continuing_time_limit.py
"""

from collections import deque
import gym
import numpy as np
from gym import spaces
import cv2
cv2.ocl.setUseOpenCL(False)


class ContinuingTimeLimit(gym.Wrapper):
    """TimeLimit wrapper for continuing environments.
    This is similar gym.wrappers.TimeLimit, which sets a time limit for
    each episode, except that done=False is returned and that
    info['needs_reset'] is set to True when past the limit.
    Code that calls env.step is responsible for checking the info dict, the
    fourth returned value, and resetting the env if it has the 'needs_reset'
    key and its value is True.
    Args:
        env (gym.Env): Env to wrap.
        max_episode_steps (int): Maximum number of timesteps during an episode,
            after which the env needs a reset.
    """

    def __init__(self, env, max_episode_steps):
        super(ContinuingTimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps

        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1

        if self._max_episode_steps <= self._elapsed_steps:
            info["needs_reset"] = True

        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()
    
    
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.

        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(
                1, self.noop_max + 1
            )  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, info = self.env.step(self.noop_action)
            if done or info.get("needs_reset", False):
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for envs that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(1)
        if done or info.get("needs_reset", False):
            self.env.reset(**kwargs)
        obs, _, done, info = self.env.step(2)
        if done or info.get("needs_reset", False):
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game end.

        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.needs_real_reset = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.needs_real_reset = done or info.get("needs_reset", False)
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few
            # frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.

        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.needs_real_reset:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done or info.get("needs_reset", False):
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, channel_order="hwc"):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            "hwc": (self.height, self.width, 1),
            "chw": (1, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        return frame.reshape(self.observation_space.low.shape)


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, channel_order="hwc"):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.stack_axis = {"hwc": 2, "chw": 0}[channel_order]
        orig_obs_space = env.observation_space
        low = np.repeat(orig_obs_space.low, k, axis=self.stack_axis)
        high = np.repeat(orig_obs_space.high, k, axis=self.stack_axis)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=orig_obs_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames), stack_axis=self.stack_axis)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Divide frame values by 255.0 and return them as np.float32.

    Especially, when the original env.observation_space is np.uint8,
    this wrapper converts frame values into [0.0, 1.0] of dtype np.float32.
    """

    def __init__(self, env):
        assert isinstance(env.observation_space, spaces.Box)
        gym.ObservationWrapper.__init__(self, env)

        self.scale = 255.0

        orig_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=self.observation(orig_obs_space.low),
            high=self.observation(orig_obs_space.high),
            dtype=np.float32,
        )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / self.scale


class LazyFrames(object):
    """Array-like object that lazily concat multiple frames.

    This object ensures that common frames between the observations are only
    stored once.  It exists purely to optimize memory usage which can be huge
    for DQN's 1M frames replay buffers.

    This object should only be converted to numpy array before being passed to
    the model.

    You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames, stack_axis=2):
        self.stack_axis = stack_axis
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=self.stack_axis)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FlickerFrame(gym.ObservationWrapper):
    """Stochastically flicker frames."""

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)

    def observation(self, observation):
        if self.unwrapped.np_random.rand() < 0.5:
            return np.zeros_like(observation)
        else:
            return observation


def make_atari(env_id, max_frames=30 * 60 * 60):
    env = gym.make(env_id)
    assert "NoFrameskip" in env.spec.id
    assert isinstance(env, gym.wrappers.TimeLimit)
    # Unwrap TimeLimit wrapper because we use our own time limits
    env = env.env
    if max_frames:
        env = ContinuingTimeLimit(env, max_episode_steps=max_frames)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    return env


def wrap_deepmind(
    env,
    episode_life=True,
    clip_rewards=True,
    frame_stack=True,
    scale=False,
    fire_reset=False,
    channel_order="chw",
    flicker=False,
    permute_obs=False,
    patch_size=6,
    rand_zero_out_ratio=0.0,
):
    """Configure environment for DeepMind-style Atari."""
    if episode_life:
        env = EpisodicLifeEnv(env)
    if fire_reset and "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = PermuteWarpFrame(
        env=env,
        channel_order=channel_order,
        permute_obs=permute_obs,
        rand_zero_out_ratio=rand_zero_out_ratio,
        patch_size=patch_size,
    )
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if flicker:
        env = FlickerFrame(env)
    if frame_stack:
        env = FrameStack(env, 4, channel_order=channel_order)
    return env


class PermuteWarpFrame(gym.ObservationWrapper):
    def __init__(self,
                 env,
                 channel_order="hwc",
                 permute_obs=True,
                 rand_zero_out_ratio=0.,
                 patch_size=6):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        shape = {
            "hwc": (self.height, self.width, 1),
            "chw": (1, self.height, self.width),
        }
        self.observation_space = spaces.Box(
            low=0, high=255, shape=shape[channel_order], dtype=np.uint8
        )
        self.original_obs = None
        self.shuffled_obs = None
        self.gray_obs = None
        self.permute_obs = permute_obs
        self.rand_zero_out_ratio = rand_zero_out_ratio
        self.patch_size = patch_size
        self.num_patches = (84 // self.patch_size) ** 2
        self.perm_ix = np.arange(self.num_patches)
        self.zero_out_ix = np.arange(self.num_patches)
        self.np_random = np.random.RandomState(0)
        self.step_cnt = 0
        self.patch_to_keep_ix = None

    def observation(self, frame):
        self.original_obs = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        self.shuffled_obs = self.shuffle_patches(self.original_obs)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self.width, self.height), interpolation=cv2.INTER_AREA
        )
        frame[:10] = 0   # Zero-out the score area.
        frame = self.shuffle_patches(frame)
        self.gray_obs = frame
        return frame.reshape(self.observation_space.low.shape)

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return self.env.seed(seed)

    def reset(self, **kwargs):
        self.patch_to_keep_ix = None
        self.perm_ix = np.arange(self.num_patches)
        self.zero_out_ix = np.arange(self.num_patches)
        self.np_random.shuffle(self.zero_out_ix)
        # print(self.zero_out_ix)
        if self.permute_obs:
            self.np_random.shuffle(self.perm_ix)
        self.step_cnt = 0
        return super(PermuteWarpFrame, self).reset(**kwargs)

    def step(self, action):
        self.step_cnt += 1
        obs, reward, done, info = super(PermuteWarpFrame, self).step(action)
        return obs, reward, done, info

    def shuffle_patches(self, obs):
        shuffled_obs = np.zeros_like(obs)
        num_patches_per_dim = 84 // self.patch_size
        num_patches_to_zero_out = int(
            (num_patches_per_dim ** 2) * self.rand_zero_out_ratio)
        patch_ix_to_zero_out = self.zero_out_ix[:num_patches_to_zero_out]
        self.patch_to_keep_ix = self.zero_out_ix[num_patches_to_zero_out:]
        for pstart_r in range(num_patches_per_dim):
            for pstart_c in range(num_patches_per_dim):
                ix = pstart_r * num_patches_per_dim + pstart_c
                shuffled_ix = self.perm_ix[ix]
                spstart_r = shuffled_ix // num_patches_per_dim
                spstart_c = shuffled_ix % num_patches_per_dim
                if ix in patch_ix_to_zero_out:
                    pass
                else:
                    sr_ss = pstart_r * self.patch_size
                    sc_ss = pstart_c * self.patch_size
                    sr_ee = sr_ss + self.patch_size
                    sc_ee = sc_ss + self.patch_size
                    r_ss = spstart_r * self.patch_size
                    c_ss = spstart_c * self.patch_size
                    r_ee = r_ss + self.patch_size
                    c_ee = c_ss + self.patch_size
                    shuffled_obs[sr_ss:sr_ee, sc_ss:sc_ee] = (
                        obs[r_ss:r_ee, c_ss:c_ee]
                    )
        return shuffled_obs

    def render(self, mode='human', **kwargs):
        if mode == 'human':
            screen = np.concatenate(
                [self.original_obs, self.shuffled_obs], axis=1)
            resized_screen = cv2.resize(screen, (400, 200))[:, :, ::-1]
            cv2.imshow('Puzzle Pong', resized_screen)
            cv2.waitKey(1)
        else:
            return self.env.render(mode)
