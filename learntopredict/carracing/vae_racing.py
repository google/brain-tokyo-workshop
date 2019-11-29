import numpy as np
import gym

from scipy.misc import imresize as resize
from gym.spaces.box import Box
from gym.envs.box2d.car_racing import CarRacing

from vae.vae import ConvVAE

from config import games

from model import SimpleWorldModel

import json

#import imageio

SCREEN_X = 64
SCREEN_Y = 64

FRAME_STACK = 16

TIME_LIMIT = 1000

MU_MODE = True # use mu rather than z

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

def _compress_frames(frames, vae):
  frames = np.array(frames).reshape(1, FRAME_STACK, 64*64)
  frames = np.swapaxes(frames, 1, 2).reshape(1, 64, 64, FRAME_STACK)
  mu, logvar = vae.encode_mu_logvar(frames/255.0)
  mu = mu[0]
  logvar = logvar[0]
  s = logvar.shape
  z = mu + np.exp(logvar/2.0) * np.random.randn(*s)
  return z

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
    self.real_z = None

  def _reset(self):
    self._internal_counter = 0
    self._has_rendered = False
    self.real_frame = None
    self.real_z = None
    return super(VAERacing, self)._reset()

  def _render(self, mode='human', close=False):
    if mode == 'human' or mode == 'rgb_array':
      self._has_rendered = True
    return super(VAERacing, self)._render(mode=mode, close=close)

  def _step(self, action):

    if not self._has_rendered:
      self._render("rgb_array")
      self._has_rendered = False

    if action is not None:
      action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
      action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
      action[1] = (action[1]+1.0) / 2.0
      action[2] = _clip(action[2])

    obs, reward, done, _ = super(VAERacing, self)._step(action)

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
      self.real_z = mu
      return mu, reward, done, {}
    self.real_z = z
    return z, reward, done, {}

class VAERacingWorld(CarRacing):
  def __init__(self, full_episode=False, pure_world=False):
    super(VAERacingWorld, self).__init__()
    self._internal_counter = 0
    self.z_size = games['vae_racing'].input_size
    self.vae = ConvVAE(batch_size=1, z_size=self.z_size, gpu_mode=False, is_training=False, reuse=True)
    self.vae.load_json('vae/vae_'+str(self.z_size)+'.json')
    self.full_episode = full_episode
    if pure_world:
      high = np.array([np.inf] * 10)
    else:
      high = np.array([np.inf] * (self.z_size+10))
    self.observation_space = Box(-high, high)
    self._has_rendered = False
    self.real_frame = None
    self.world_model = SimpleWorldModel(obs_size=16, action_size=3, hidden_size=10)
    world_model_path = "./log/learn_vae_racing.cma.4.64.best.json"
    self.world_model.load_model(world_model_path)
    self.pure_world_mode = pure_world

  def _reset(self):
    self._internal_counter = 0
    self._has_rendered = False
    self.real_frame = None
    return super(VAERacingWorld, self)._reset()

  def _render(self, mode='human', close=False):
    if mode == 'human' or mode == 'rgb_array':
      self._has_rendered = True
    return super(VAERacingWorld, self)._render(mode=mode, close=close)

  def _step(self, action):

    if not self._has_rendered:
      self._render("rgb_array")
      self._has_rendered = False

    old_action = [0, 0, 0]

    if action is not None:
      old_action = np.copy(action)
      action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
      action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
      action[1] = (action[1]+1.0) / 2.0
      action[2] = _clip(action[2])

    obs, reward, done, _ = super(VAERacingWorld, self)._step(action)

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
      z = mu

    self.world_model.predict_next_obs(z, old_action)

    if self.pure_world_mode:
      z = np.copy(self.world_model.hidden_state)
    else:
      z = np.concatenate([z, self.world_model.hidden_state], axis=0)

    return z, reward, done, {}

# useful for discrete actions
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print('error with sampling ensemble')
  return -1

# useful for discrete actions
def sample(p):
  #return np.argmax(np.random.multinomial(1, p))
  return get_pi_idx(np.random.rand(), p)

class VAERacingStack(CarRacing):
  def __init__(self, full_episode=False, discrete_mode=False):
    super(VAERacingStack, self).__init__()
    self._internal_counter = 0
    self.z_size = games['vae_racing_stack'].input_size
    self.vae = ConvVAE(batch_size=1, z_size=self.z_size, num_channel=FRAME_STACK, gpu_mode=False, is_training=False, reuse=True)
    self.vae.load_json('vae/vae_stack_'+str(FRAME_STACK)+'.json')
    self.full_episode = full_episode
    high = np.array([np.inf] * self.z_size)
    self.observation_space = Box(-high, high)
    self.cumulative_frames = None
    self._has_rendered = False
    self.discrete_mode = discrete_mode

  def _get_image(self, z, cumulative_frames):
    large_img = np.zeros((64*2, 64*FRAME_STACK))

    # decode the latent vector
    if z is not None:
      img = self.vae.decode(z.reshape(1, self.z_size)) * 255.0
      img = np.round(img).astype(np.uint8)
      img = img.reshape(64, 64, FRAME_STACK)
      for i in range(FRAME_STACK):
        large_img[64:, i*64:(i+1)*64] = img[:, :, i]

    if len(cumulative_frames) == FRAME_STACK:
      for i in range(FRAME_STACK):
        large_img[:64, i*64:(i+1)*64] = cumulative_frames[i]

    large_img = large_img.astype(np.uint8)

    return large_img

  def _reset(self):
    self._internal_counter = 0
    self.cumulative_frames = None
    self._has_rendered = False
    return super(VAERacingStack, self)._reset()

  def _render(self, mode='human', close=False):
    if mode == 'human' or mode == 'rgb_array':
      self._has_rendered = True
    return super(VAERacingStack, self)._render(mode=mode, close=close)

  def _step(self, action):

    if not self._has_rendered:
      self._render("rgb_array")
      self._has_rendered = False

    if action is not None:
      if not self.discrete_mode:
        action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
        action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
        action[1] = (action[1]+1.0) / 2.0
        action[2] = _clip(action[2])
      else:
        '''
        in discrete setting:
        if action[0] is the highest, then agent does nothing
        if action[1] is the highest, then agent hits the pedal
        if -action[1] is the highest, then agent hits the brakes
        if action[2] is the highest, then agent turns left
        if action[3] is the highest, then agent turns right
        '''
        logits = [_clip((action[0]+1.0), hi=+2.0),
          _clip(action[1]),
          _clip(-action[1]),
          _clip(action[2]),
          _clip(-action[2])]
        probs = softmax(logits)
        #chosen_action = np.argmax(logits)
        chosen_action = sample(probs)

        a = np.array( [0.0, 0.0, 0.0] )

        if chosen_action == 1: a[1] = +1.0 # up
        if chosen_action == 2: a[2] = +0.8 # down: 0.8 as recommended by the environment's built-in demo
        if chosen_action == 3: a[0] = -1.0 # left
        if chosen_action == 4: a[0] = +1.0 # right

        action = a
        #print("chosen_action", chosen_action, action)

    obs, reward, done, _ = super(VAERacingStack, self)._step(action)

    if self.cumulative_frames is not None:
      self.cumulative_frames.pop(0)
      self.cumulative_frames.append(_process_frame_green(obs))
    else:
      self.cumulative_frames = [_process_frame_green(obs)] * FRAME_STACK

    self.z = z = _compress_frames(self.cumulative_frames, self.vae)

    if self.full_episode:
      return z, reward, False, {}

    self._internal_counter += 1
    if self._internal_counter > TIME_LIMIT:
      done = True

    #img = self._get_image(self.z, self.cumulative_frames)
    #imageio.imwrite('dump/'+('%0*d' % (4, self._internal_counter))+'.png', img)

    return z, reward, done, {}


