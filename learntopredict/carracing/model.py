import numpy as np
import random
# I implemented Schmidhuber's "Compressed Network Search" but didn't use it.
# ndded for the compress/decompress functions.
#from scipy.fftpack import dct
import json
import sys
import config
from env import make_env
import time
import os

from gym.wrappers import Monitor

from nn import sigmoid, relu, passthru, softmax, sample, RNNModel

np.set_printoptions(precision=3, threshold=20, linewidth=200)

PEEK_PROB = 0.0
SIMPLE_MODE = False

with open("peek_prob.json") as f:    
  PEEK_PROB = json.load(f)
with open("simple_mode.json") as f:    
  SIMPLE_MODE = json.load(f)

final_mode = False
render_mode = True
RENDER_DELAY = False
record_video = False
MEAN_MODE = False

record_rgb = False

if record_rgb:
  import imageio

def compress_2d(w, shape=None):
  s = w.shape
  if shape:
    s = shape
  c = dct(dct(w, axis=0, type=2, norm='ortho'), axis=1, type=2, norm='ortho')
  return c[0:s[0], 0:s[1]]

def decompress_2d(c, shape):
  c_out = np.zeros(shape)
  c_out[0:c.shape[0], 0:c.shape[1]] = c
  w = dct(dct(c_out.T, type=3, norm='ortho').T, type=3, norm='ortho')
  return w

def compress_1d(w, shape=None, axis=0):
  s = w.shape
  if shape:
    s = shape
  c = dct(w, axis=axis, type=2, norm='ortho')
  return c[0:s[0], 0:s[1]]

def decompress_1d(c, shape, axis=0):
  c_out = np.zeros(shape)
  c_out[0:c.shape[0], 0:c.shape[1]] = c
  w = dct(c_out, axis=axis, type=3, norm='ortho')
  return w

def make_model(game):
  # can be extended in the future.
  if game.rnn_mode:
    model = RNNModel(game)
  elif game.experimental_mode:
    model = CustomModel(game)
  else:
    model = Model(game)
  return model

# LSTM in a few lines of numpy
class LSTMCell:
  '''Numpy LSTM cell used for inference only.'''
  def __init__(self, input_size, weight, bias, forget_bias=1.0, dropout_keep_prob=0.5, train_mode=True):
    self.input_size=input_size
    self.W_full=weight # np.concatenate((Wxh, Whh), axis=0)
    self.bias=bias
    self.forget_bias=1.0
    self.dropout_keep_prob = dropout_keep_prob
    self.train_mode = train_mode
    self.hidden_size = int(bias.shape[0]/4)

  def __call__(self, x, h, c):

    concat = np.concatenate((x, h), axis=1)
    hidden = np.matmul(concat, self.W_full)+self.bias

    i, g, f, o = np.split(hidden, 4, axis=1)

    i = sigmoid(i)
    g = np.tanh(g)
    f = sigmoid(f+self.forget_bias)
    o = sigmoid(o)

    if self.train_mode:
      mask = np.array(np.random.rand(self.hidden_size) < self.dropout_keep_prob).astype(np.int)
      d_g = np.multiply(mask, g)
    else:
      d_g = self.dropout_keep_prob * g
    
    new_c = np.multiply(c, f) + np.multiply(d_g, i)
    new_h = np.multiply(np.tanh(new_c), o)

    return new_h, new_c

  def set_dropout_keep_prob(self, dropout_keep_prob=0.5, train_mode=True):
    self.dropout_keep_prob = dropout_keep_prob
    self.train_mode = train_mode

class RNNWorldModel:
  ''' deterministic LSTM model for cart-pole swing up task '''
  def __init__(self, obs_size=5, action_size=1, hidden_size=20, dropout_keep_prob=0.5, train_mode=True, predict_future=False):
    self.obs_size = obs_size
    self.action_size = action_size
    self.hidden_size = hidden_size

    self.predict_future = predict_future
    if self.predict_future:
      self.shapes = [ (self.obs_size + self.action_size + self.hidden_size, 4*self.hidden_size), # LSTM weights
                      (self.hidden_size, 2) # predict next observation
                    ]
    else:
      self.shapes = [ (self.obs_size + self.action_size + self.hidden_size, 4*self.hidden_size), # LSTM weights
                    ]

    self.dropout_keep_prob = dropout_keep_prob
    self.train_mode = train_mode

    self.weight = []
    self.bias = []
    self.param_count = 0

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      idx += 1
    
    self.init_h = np.zeros((1, self.hidden_size))
    self.init_c = np.zeros((1, self.hidden_size))
    self.h = self.init_h
    self.c = self.init_c
    self.param_count += 2*self.hidden_size
    
    self.lstm = LSTMCell(self.obs_size-1 + self.action_size, self.weight[0], self.bias[0], dropout_keep_prob=self.dropout_keep_prob, train_mode=train_mode)

  def set_dropout_keep_prob(self, dropout_keep_prob=0.5, train_mode=True):
    self.dropout_keep_prob = dropout_keep_prob
    self.train_mode = train_mode
    self.lstm.set_dropout_keep_prob(dropout_keep_prob=self.dropout_keep_prob, train_mode=train_mode)

  def get_state(self):
    return np.concatenate([self.h, self.c], axis=1)

  def reset_state(self):
    self.h = self.init_h
    self.c = self.init_c

  def update(self, obs, action):
    total_obs = np.concatenate([obs.flatten(), action.flatten()]).reshape((1, self.obs_size+self.action_size))
    self.h, self.c = self.lstm(total_obs, self.h, self.c)

  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)
      pointer += s
    # rnn states
    s = self.hidden_size
    self.init_h = model_params[pointer:pointer+s].reshape((1, self.hidden_size))
    pointer += s
    self.init_c = model_params[pointer:pointer+s].reshape((1, self.hidden_size))

    self.reset_state()
    self.lstm = LSTMCell(self.obs_size + self.action_size, self.weight[0], self.bias[0], dropout_keep_prob=self.dropout_keep_prob, train_mode=self.train_mode)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

class RNNModel:
  ''' learning the best feed forward model for cartpole-swingup '''
  def __init__(self, game):
    self.env_name = game.env_name

    self.layer_1 = game.layers[0]
    self.layer_2 = game.layers[1]
    self.world_hidden_size = self.layer_1
    self.agent_hidden_size = self.layer_2

    self.rnn_mode = True
    self.experimental_mode = False

    self.input_size = game.input_size
    self.output_size = game.output_size

    self.render_mode = False
    self.dropout_keep_prob = 1.0

    self.world_model = RNNWorldModel(obs_size=self.input_size, action_size=self.output_size, hidden_size=self.world_hidden_size, dropout_keep_prob=self.dropout_keep_prob, predict_future=False)
    self.agent = Agent(layer_1=self.agent_hidden_size, layer_2=0, input_size=self.input_size+self.world_hidden_size*2, output_size=self.output_size)

    self.param_count = self.world_model.param_count + self.agent.param_count

  def reset(self):
    ''' solve for best weights for agent, aka the inner-loop '''
    self.world_model.reset_state()

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def get_action(self, obs):
    total_obs = np.concatenate([np.array(obs).flatten(), self.world_model.get_state().flatten()])
    action = self.agent.get_action(total_obs)

    self.world_model.update(obs, action)

    return action

  def set_model_params(self, model_params):
    world_model_params = model_params[:self.world_model.param_count]
    agent_model_params = model_params[self.world_model.param_count:self.world_model.param_count+self.agent.param_count]
    self.world_model.set_model_params(world_model_params)
    self.agent.set_model_params(agent_model_params)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

class Agent:
  ''' simple feedforward model to act on world model's hidden state '''
  def __init__(self, layer_1=10, layer_2=5, input_size=5+20*2, output_size=1):
    self.layer_1 = layer_1
    self.layer_2 = layer_2
    self.input_size = input_size #
    self.output_size = output_size # action space
    if layer_2 == 0:
      self.shapes = [ (self.input_size, self.layer_1),
                      (self.layer_1, self.output_size)]
    else:
      self.shapes = [ (self.input_size, self.layer_1),
                      (self.layer_1, self.layer_2),
                      (self.layer_2, self.output_size)]

    self.activations = [np.tanh, np.tanh, np.tanh] # assumption that output is bounded between -1 and 1 (pls chk!)

    self.weight = []
    self.bias = []
    self.param_count = 0

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      idx += 1

  def get_action(self, x):
    h = np.array(x).flatten()
    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      h = self.activations[i](h)
    return h

  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)
      pointer += s

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

def _clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

class SimpleWorldModel:
  ''' deterministic LSTM model for cart-pole swing up task '''
  def __init__(self, obs_size=16, action_size=3, hidden_size=10):
    self.obs_size = obs_size
    self.action_size = action_size
    self.hidden_size = hidden_size

    self.shapes = [ (self.obs_size + self.action_size, self.hidden_size),
                    (self.hidden_size, self.obs_size)]

    self.weight = []
    self.bias = []
    self.param_count = 0

    self.dt = 1.0 / 50.0 # 50 fps

    self.hidden_state = np.zeros(self.hidden_size)

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      idx += 1

  def reset(self):
    self.hidden_state = np.zeros(self.hidden_size)

  def predict_next_obs(self, obs, action):

    obs = np.array(obs).flatten()

    new_action = np.array( [0.0, 0.0, 0.0] )

    new_action[0] = _clip(action[0], lo=-1.0, hi=+1.0)
    new_action[1] = _clip(action[1], lo=-1.0, hi=+1.0)
    new_action[1] = (action[1]+1.0) / 2.0
    new_action[2] = _clip(action[2])

    h = np.concatenate([obs, new_action.flatten()])

    activations = [np.tanh, passthru]

    num_layers = 2

    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      h = activations[i](h)
      if (i == 0): # save the hidden state
        self.hidden_state = h.flatten()

    prediction = obs + h.flatten() * self.dt # residual

    return prediction

  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])
      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)
      pointer += s

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

class CustomModel:
  ''' learning the best feed forward model for vae_racing '''
  def __init__(self, game):
    self.output_noise = game.output_noise
    self.env_name = game.env_name
    self.world_hidden_size = game.layers[0]
    self.agent_hidden_size = game.layers[1]
    self.rnn_mode = False # in the future will be useful
    self.experimental_mode = True
    self.peek_prob = PEEK_PROB
    self.simple_mode = SIMPLE_MODE
    self.peek_next = 1
    self.peek = 1
    self.counter = 0

    self.input_size = game.input_size # observation size
    self.output_size = game.output_size # action size

    self.render_mode = False

    self.world_model = SimpleWorldModel(obs_size=self.input_size, action_size=self.output_size, hidden_size=self.world_hidden_size)
    agent_input_size = self.input_size+self.world_hidden_size
    if self.simple_mode:
      agent_input_size = self.input_size
    self.agent = Agent(layer_1=self.agent_hidden_size, layer_2=0, input_size=agent_input_size, output_size=self.output_size)

    self.param_count = self.world_model.param_count + self.agent.param_count
    self.prev_action = np.zeros(self.output_size)
    self.prev_prediction = None

    #self.temp_obs = np.zeros(16)
    #self.temp_predict = np.zeros(16)

  def reset(self):
    self.prev_prediction = None
    self.peek_next = 1
    self.peek = 1
    self.counter = 0
    self.world_model.reset()

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def get_action(self, real_obs, t=0, mean_mode=False):
    obs = real_obs
    use_prediction = False

    self.counter += 1 # for tracking frames in case we want to dump out rgb images

    if (self.prev_prediction is not None) and (self.peek_next == 0):
      obs = self.prev_prediction
      use_prediction = True

    if record_rgb:
      video_path = "learning_vae_racing_dump"
      video_path_orig = "learning_vae_racing_dump/orig"
      if not os.path.exists(video_path):
        os.makedirs(video_path)
      if not os.path.exists(video_path_orig):
        os.makedirs(video_path_orig)
      img = self.env.vae.decode(obs.reshape(1, self.env.z_size)) * 255.
      img = np.round(img).astype(np.uint8)
      img = img.reshape(64, 64, 3)
      orig_img = img
      real_img = np.round(self.env.real_frame * 255.).astype(np.uint8).reshape(64, 64, 3)
      extension = ".real.png"
      if use_prediction:
        extension = ".predict.png"
        #img = 255-img
      video_extension = ".truth.png"
      #total_img = np.concatenate([real_img, img], axis=1)
      imageio.imwrite(os.path.join(video_path_orig,(format(self.counter, "05d")+extension)), orig_img)
      imageio.imwrite(os.path.join(video_path,("b_"+format(self.counter, "05d")+video_extension)), real_img)

      recon_img = self.env.vae.decode(self.env.real_z.reshape(1, self.env.z_size)) * 255.
      recon_img = np.round(recon_img).astype(np.uint8)
      recon_img = recon_img.reshape(64, 64, 3)
      imageio.imwrite(os.path.join(video_path,("a_"+format(self.counter, "05d")+video_extension)), recon_img)

    #if (self.prev_prediction is not None) and render_mode:
      #print("use_prediction", use_prediction)
      #print("rms change real_obs", np.sqrt(np.mean(np.square(np.array(real_obs) - self.temp_obs))))
      #print("rms change prediction", np.sqrt(np.mean(np.square(np.array(self.prev_prediction) - self.temp_predict))))
      #self.temp_obs = real_obs
      #self.temp_predict = self.prev_prediction

    if self.simple_mode:
      agent_obs = obs
    else:
      prev_hidden = self.world_model.hidden_state.flatten()
      agent_obs = np.concatenate([obs.flatten(), prev_hidden]) # use previous hidden state
    action = self.agent.get_action(agent_obs)

    self.peek = self.peek_next
    self.peek_next = 0
    if (np.random.rand() < self.peek_prob):
      self.peek_next = 1

    self.prev_prediction = self.world_model.predict_next_obs(obs, action) # update hidden state, and predict next frame
    return action

  def set_model_params(self, model_params):
    world_params = model_params[:self.world_model.param_count]
    agent_params = model_params[self.world_model.param_count:self.world_model.param_count+self.agent.param_count]

    assert len(world_params) == self.world_model.param_count, "inconsistent world model params"
    assert len(agent_params) == self.agent.param_count, "inconsistent agent params"
    self.world_model.set_model_params(world_params)
    self.agent.set_model_params(agent_params)

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

class Model:
  ''' simple feedforward model '''
  def __init__(self, game):
    self.output_noise = game.output_noise
    self.env_name = game.env_name
    self.layer_1 = game.layers[0]
    self.layer_2 = game.layers[1]
    self.rnn_mode = False # in the future will be useful
    self.experimental_mode = False
    self.time_input = 0 # use extra sinusoid input
    self.sigma_bias = game.noise_bias # bias in stdev of output
    self.sigma_factor = 0.5 # multiplicative in stdev of output
    if game.time_factor > 0:
      self.time_factor = float(game.time_factor)
      self.time_input = 1
    self.input_size = game.input_size
    self.output_size = game.output_size
    if self.layer_2 > 0:
      self.shapes = [ (self.input_size + self.time_input, self.layer_1),
                      (self.layer_1, self.layer_2),
                      (self.layer_2, self.output_size)]
    elif self.layer_1 == 0 and self.layer_2 == 0:
      self.shapes = [ (self.input_size + self.time_input, self.output_size) ]
    elif self.layer_1 > 0 and self.layer_2 == 0:
      self.shapes = [ (self.input_size + self.time_input, self.layer_1),
                      (self.layer_1, self.output_size)]
    else:
      assert False, "invalid layer_2"

    self.sample_output = False
    if game.activation == 'relu':
      self.activations = [relu, relu, passthru]
    elif game.activation == 'sigmoid':
      self.activations = [np.tanh, np.tanh, sigmoid]
    elif game.activation == 'softmax':
      self.activations = [np.tanh, np.tanh, softmax]
      self.sample_output = True
    elif game.activation == 'passthru':
      self.activations = [np.tanh, np.tanh, passthru]
    else:
      self.activations = [np.tanh, np.tanh, np.tanh]

    self.weight = []
    self.bias = []
    self.bias_log_std = []
    self.bias_std = []
    self.param_count = 0

    idx = 0
    for shape in self.shapes:
      self.weight.append(np.zeros(shape=shape))
      self.bias.append(np.zeros(shape=shape[1]))
      self.param_count += (np.product(shape) + shape[1])
      if self.output_noise[idx]:
        self.param_count += shape[1]
      log_std = np.zeros(shape=shape[1])
      self.bias_log_std.append(log_std)
      out_std = np.exp(self.sigma_factor*log_std + self.sigma_bias)
      self.bias_std.append(out_std)
      idx += 1

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def get_action(self, x, t=0, mean_mode=False):
    # if mean_mode = True, ignore sampling.
    h = np.array(x).flatten()
    if self.time_input == 1:
      time_signal = float(t) / self.time_factor
      h = np.concatenate([h, [time_signal]])
    num_layers = len(self.weight)
    for i in range(num_layers):
      w = self.weight[i]
      b = self.bias[i]
      h = np.matmul(h, w) + b
      if (self.output_noise[i] and (not mean_mode)):
        out_size = self.shapes[i][1]
        out_std = self.bias_std[i]
        output_noise = np.random.randn(out_size)*out_std
        h += output_noise
      h = self.activations[i](h)

    if self.sample_output:
      h = sample(h)

    return h

  def set_model_params(self, model_params):
    pointer = 0
    for i in range(len(self.shapes)):
      w_shape = self.shapes[i]
      b_shape = self.shapes[i][1]
      s_w = np.product(w_shape)
      s = s_w + b_shape
      chunk = np.array(model_params[pointer:pointer+s])

      self.weight[i] = chunk[:s_w].reshape(w_shape)
      self.bias[i] = chunk[s_w:].reshape(b_shape)

      '''
      s_b = np.product(b_shape) # works with legacy world models Z_H_HIDDEN mode format
      self.weight[i] = chunk[s_b:].reshape(w_shape)
      self.bias[i] = chunk[:s_b].reshape(b_shape)
      '''

      pointer += s
      if self.output_noise[i]:
        s = b_shape
        self.bias_log_std[i] = np.array(model_params[pointer:pointer+s])
        self.bias_std[i] = np.exp(self.sigma_factor*self.bias_log_std[i] + self.sigma_bias)
        if self.render_mode:
          print("bias_std, layer", i, self.bias_std[i])
        pointer += s

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

def evaluate(model):
  # run 100 times and average score, according to the reles.
  model.env.seed(0)
  total_reward = 0.0
  N = 100
  for i in range(N):
    reward, t = simulate(model, train_mode=False, render_mode=False, num_episode=1)
    total_reward += reward[0]
  return (total_reward / float(N))

def compress_input_dct(obs):
  new_obs = np.zeros((8, 8))
  for i in range(obs.shape[2]):
    new_obs = +compress_2d(obs[:, :, i] / 255., shape=(8, 8))
  new_obs /= float(obs.shape[2])
  return new_obs.flatten()

def simulate(model, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):

  reward_list = []
  t_list = []

  is_biped = (model.env_name.find("BipedalWalker") >= 0)

  orig_mode = True  # hack for bipedhard's reward augmentation during training (set to false for hack)
  if is_biped:
    orig_mode = False

  dct_compress_mode = False

  max_episode_length = 1000

  if train_mode and max_len > 0:
    if max_len < max_episode_length:
      max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    model.env.seed(seed)

  for episode in range(num_episode):

    if model.rnn_mode:
      model.reset()

    if model.experimental_mode:
      model.reset()

    obs = model.env.reset()
    if dct_compress_mode and obs is not None:
      obs = compress_input_dct(obs)

    if obs is None:
      obs = np.zeros(model.input_size)

    total_reward = 0.0
    stumbled = False # hack for bipedhard's reward augmentation during training. turned off.
    reward_threshold = 300 # consider we have won if we got more than this

    num_glimpse = 0

    for t in range(max_episode_length):

      if render_mode:
        model.env.render("human")
        if RENDER_DELAY:
          time.sleep(0.01)

      if model.rnn_mode:
        action = model.get_action(obs)
      else:
        if MEAN_MODE:
          action = model.get_action(obs, t=t, mean_mode=(not train_mode))
        else:
          action = model.get_action(obs, t=t, mean_mode=False)

      prev_obs = obs

      obs, reward, done, info = model.env.step(action)

      if model.experimental_mode: # augment reward with prob
        num_glimpse += model.peek

      if dct_compress_mode:
        obs = compress_input_dct(obs)

      if train_mode and reward == -100 and (not orig_mode):
        # hack for bipedhard's reward augmentation during training. turned off.
        reward = 0
        stumbled = True

      if (render_mode):
        pass
        #print("action", action, "step reward", reward)
        #print("step reward", reward)
      total_reward += reward

      if done:
        if train_mode and (not stumbled) and (total_reward > reward_threshold) and (not orig_mode):
           # hack for bipedhard's reward augmentation during training. turned off.
          total_reward += 100
        break

    if render_mode:
      print("reward", total_reward, "timesteps", t)
      if model.experimental_mode:
        print("percent glimpse", float(num_glimpse)/float(t+1.0))
    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():

  global RENDER_DELAY

  assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

  gamename = sys.argv[1]

  if gamename.startswith("bullet"):
    RENDER_DELAY = True

  use_model = False

  game = config.games[gamename]

  if len(sys.argv) > 2:
    use_model = True
    filename = sys.argv[2]
    print("filename", filename)

  the_seed = 0
  if len(sys.argv) > 3:
    the_seed = int(sys.argv[3])
    print("seed", the_seed)

  model = make_model(game)
  print('model size', model.param_count)

  model.make_env(render_mode=render_mode)

  if use_model:
    model.load_model(filename)
  else:
    params = model.get_random_model_params(stdev=0.5)
    model.set_model_params(params)

  if final_mode:
    rewards = []

    for i in range(100):
      reward, steps_taken = simulate(model, train_mode=False, render_mode=False, num_episode=1, seed=the_seed+i)
      print(i, reward)
      rewards.append(reward[0])
    print("seed", the_seed, "average_reward", np.mean(rewards), "standard_deviation", np.std(rewards))
  else:
    if record_video:
      model.env = Monitor(model.env, directory='/tmp/'+gamename,video_callable=lambda episode_id: True, write_upon_reset=True, force=True)
    for i in range(1):
      reward, steps_taken = simulate(model,
        train_mode=False, render_mode=render_mode, num_episode=1, seed=the_seed+i)
      print ("terminal reward", reward, "average steps taken", np.mean(steps_taken)+1)

if __name__ == "__main__":
  main()
