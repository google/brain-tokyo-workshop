# neural network functions and classes

import numpy as np
import random
import json
import cma
from es import SimpleGA, CMAES, PEPG, OpenES
from env import make_env

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def relu(x):
  return np.maximum(x, 0)

def passthru(x):
  return x

# useful for discrete actions
def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

# useful for discrete actions
def sample(p):
  return np.argmax(np.random.multinomial(1, p))

"""
learning the model
"""

class RNNCell:
  def __init__(self, input_size, weight, bias):
    self.input_size=input_size
    self.weight = weight
    self.bias = bias
  def __call__(self, x, h):
    concat = np.concatenate((x, h), axis=1)
    hidden = np.matmul(concat, self.weight)+self.bias
    return np.tanh(hidden)

# LSTM in a few lines of numpy
class LSTMCell:
  '''Numpy LSTM cell used for inference only.'''
  def __init__(self, input_size, weight, bias, forget_bias=1.0):
    self.input_size=input_size
    self.W_full=weight # np.concatenate((Wxh, Whh), axis=0)
    self.bias=bias
    self.forget_bias=1.0

  def __call__(self, x, h, c):

    concat = np.concatenate((x, h), axis=1)
    hidden = np.matmul(concat, self.W_full)+self.bias

    i, g, f, o = np.split(hidden, 4, axis=1)

    i = sigmoid(i)
    g = np.tanh(g)
    f = sigmoid(f+self.forget_bias)
    o = sigmoid(o)
    
    new_c = np.multiply(c, f) + np.multiply(g, i)
    new_h = np.multiply(np.tanh(new_c), o)

    return new_h, new_c

class RNNModel:
  def __init__(self, game):
    self.env_name = game.env_name

    self.hidden_size = game.layers[0]

    self.layer_1 = game.layers[1]
    self.layer_2 = game.layers[2]

    self.rnn_mode = True

    self.input_size = game.input_size
    self.output_size = game.output_size

    self.render_mode = False

    self.shapes = [ (self.input_size + self.hidden_size, 1*self.hidden_size), # RNN weights
                    (self.input_size + self.hidden_size, self.layer_1),# predict actions output
                    (self.layer_1, self.output_size)] # predict actions output

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
    self.h = self.init_h
    self.param_count += 1*self.hidden_size
    
    self.rnn = RNNCell(self.input_size, self.weight[0], self.bias[0])

  def reset(self):
    self.h = self.init_h

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def get_action(self, real_obs):
    obs = real_obs.reshape(1, 3)

    # update rnn:
    #update_obs = np.concatenate([obs, action], axis=1)
    self.h = self.rnn(obs, self.h)

    # get action
    total_obs = np.concatenate([obs, self.h], axis=1)

    # calculate action using 2 layer network from output
    hidden = np.tanh(np.matmul(total_obs, self.weight[1]) + self.bias[1])
    action = np.tanh(np.matmul(hidden, self.weight[2]) + self.bias[2])

    return action[0]

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
    self.h = self.init_h
    self.rnn = RNNCell(self.input_size, self.weight[0], self.bias[0])

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

