import numpy as np
import random

import json
import sys
import config
from env import make_env
import time
import os
import ann

import argparse

from gym.wrappers import Monitor

np.set_printoptions(precision=2) 
np.set_printoptions(linewidth=160)

final_mode = True
render_mode = False

#final_mode = False; render_mode = True # VIEW: toggle with comment to view trials

RENDER_DELAY = False
record_video = False
MEAN_MODE = False

record_rgb = False

if record_rgb:
  import imageio

def make_model(game):
  # can be extended in the future.
  model = Model(game)
  return model

class Model:
  ''' simple feedforward model '''
  def __init__(self, game):

    self.env_name = game.env_name
    self.wann_file = game.wann_file
    self.input_size = game.input_size
    self.output_size = game.output_size
    self.action_select = game.action_select
    self.weight_bias = game.weight_bias

    self.wVec, self.aVec, self.wKey = ann.importNet("champions/"+self.wann_file)

    self.param_count = len(self.wKey)

    self.weights = np.zeros(self.param_count)

    self.render_mode = False

  def make_env(self, seed=-1, render_mode=False):
    self.render_mode = render_mode
    self.env = make_env(self.env_name, seed=seed, render_mode=render_mode)

  def get_action(self, x):
    # if mean_mode = True, ignore sampling.
    annOut = ann.act(self.wVec, self.aVec, self.input_size, self.output_size, x)
    action = ann.selectAct(annOut, self.action_select)

    return action

  def set_model_params(self, model_params):
    assert(len(model_params) == self.param_count)
    self.weights = np.array(model_params)
    for idx in range(self.param_count):
      key = self.wKey[idx]
      self.wVec[key] = self.weights[idx]+self.weight_bias

  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    return np.random.randn(self.param_count)*stdev

  def get_uniform_random_model_params(self, stdev=2.0):
    return np.random.rand(self.param_count)*stdev*2-stdev

  def get_single_model_params(self, weight=-1.0):
    return np.array([weight]*self.param_count)

def evaluate(model):
  # run 100 times and average score, according to the reles.
  model.env.seed(0)
  total_reward = 0.0
  N = 100
  for i in range(N):
    reward, t = simulate(model, train_mode=False, render_mode=False, num_episode=1)
    total_reward += reward[0]
  return (total_reward / float(N))

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

    obs = model.env.reset()

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

      action = model.get_action(obs)

      prev_obs = obs

      obs, reward, done, info = model.env.step(action)

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

    reward_list.append(total_reward)
    t_list.append(t)

  return reward_list, t_list

def main():

  global RENDER_DELAY

  parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
  parser.add_argument('gamename', type=str, help='robo_pendulum, robo_ant, robo_humanoid, etc.')
  parser.add_argument('-f', '--filename', type=str, help='json filename', default='none')
  parser.add_argument('-e', '--eval_steps', type=int, default=100, help='evaluate this number of step if final_mode')
  parser.add_argument('-s', '--seed_start', type=int, default=0, help='initial seed')
  parser.add_argument('-w', '--single_weight', type=float, default=-100, help='single weight parameter')
  parser.add_argument('--stdev', type=float, default=2.0, help='standard deviation for weights')
  parser.add_argument('--sweep', type=int, default=-1, help='sweep a set of weights from -2.0 to 2.0 sweep times.')
  parser.add_argument('--lo', type=float, default=-2.0, help='slow side of sweep.')
  parser.add_argument('--hi', type=float, default=2.0, help='high side of sweep.')

  args = parser.parse_args()

  assert len(sys.argv) > 1, 'python model.py gamename path_to_mode.json'

  gamename = args.gamename

  use_model = False

  game = config.games[gamename]

  filename = args.filename
  if filename != "none":
    use_model = True
    print("filename", filename)

  the_seed = args.seed_start

  model = make_model(game)
  print('model size', model.param_count)

  eval_steps = args.eval_steps
  single_weight = args.single_weight
  weight_stdev = args.stdev
  num_sweep = args.sweep
  sweep_lo = args.lo
  sweep_hi = args.hi

  model.make_env(render_mode=render_mode)

  if use_model:
    model.load_model(filename)
  else:
    if single_weight > -100:
      params = model.get_single_model_params(weight=single_weight-game.weight_bias) # REMEMBER TO UNBIAS
      print("single weight value set to", single_weight)
    else:
      params = model.get_uniform_random_model_params(stdev=weight_stdev)-game.weight_bias
    model.set_model_params(params)

  if final_mode:
    if num_sweep > 1:
      the_weights = np.arange(sweep_lo, sweep_hi+(sweep_hi-sweep_lo)/num_sweep, (sweep_hi-sweep_lo)/num_sweep)
      for i in range(len(the_weights)):
        the_weight = the_weights[i]
        params = model.get_single_model_params(weight=the_weight-game.weight_bias) # REMEMBER TO UNBIAS
        model.set_model_params(params)
        rewards = []
        for i in range(eval_steps):
          reward, steps_taken = simulate(model, train_mode=False, render_mode=False, num_episode=1, seed=the_seed+i)
          rewards.append(reward[0])
        print("weight", the_weight, "average_reward", np.mean(rewards), "standard_deviation", np.std(rewards))
    else:
      rewards = []
      for i in range(eval_steps):
        ''' random uniform params
        params = model.get_uniform_random_model_params(stdev=weight_stdev)-game.weight_bias
        model.set_model_params(params)
        '''
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
