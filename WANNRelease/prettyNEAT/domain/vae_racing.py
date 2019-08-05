import numpy as np
import gym

from scipy.misc import imresize as resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

from vae.vae import ConvVAE

from config import games

import json

#import imageio

SCREEN_X = 64
SCREEN_Y = 64

TIME_LIMIT = 1000

MU_MODE = True

def _clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

def _process_frame(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs

def _process_frame_green(frame):
  obs = frame[0:84, :, :].astype(np.float)/255.0
  obs = resize(obs, (64, 64))
  obs = ((1.0 - obs) * 255).round().astype(np.uint8)
  return obs[:, :, 1] # green channel

class VAERacing(CarRacing):
  def __init__(self, full_episode=False):
    super(VAERacing, self).__init__()
    self._internal_counter = 0
    self.z_size = games['vae_racing'].input_size
    self.vae = ConvVAE(batch_size=1, z_size=self.z_size, gpu_mode=False, is_training=False, reuse=True)
    self.vae.load_json('vae/vae_'+str(self.z_size)+'.json')
    self.full_episode = full_episode
    high = np.array([np.inf] * self.z_size)
    self.observation_space = Box(-high, high)
    self._has_rendered = False
    self.real_frame = None

  def reset(self):
    self._internal_counter = 0
    self._has_rendered = False
    self.real_frame = None
    return super(VAERacing, self).reset()

  def render(self, mode='human', close=False):
    if mode == 'human' or mode == 'rgb_array':
      self._has_rendered = True
    return super(VAERacing, self).render(mode=mode)

  def step(self, action):

    if not self._has_rendered:
      self.render("rgb_array")
      self._has_rendered = False

    if action is not None:
      action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
      action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
      action[1] = (action[1]+1.0) / 2.0
      action[2] = _clip(action[2])

    obs, reward, done, _ = super(VAERacing, self).step(action)

    result = np.copy(_process_frame(obs)).astype(np.float)/255.0
    result = result.reshape(1, 64, 64, 3)
    self.real_frame = result

    #z = self.vae.encode(result).flatten()
    mu, logvar = self.vae.encode_mu_logvar(result)
    mu = mu[0]
    logvar = logvar[0]
    s = logvar.shape
    z = mu + np.exp(logvar/2.0) * np.random.randn(*s)

    if self.full_episode:
      if MU_MODE:
        return mu, reward, False, {}
      else:
        return z, reward, False, {}

    self._internal_counter += 1
    if self._internal_counter > TIME_LIMIT:
      done = True

    if MU_MODE:
      return mu, reward, done, {}
    return z, reward, done, {}
