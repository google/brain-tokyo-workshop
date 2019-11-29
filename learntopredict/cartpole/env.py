import numpy as np
import gym

def make_env(env_name, seed=-1, render_mode=False):
  if (env_name.startswith("CartPoleSwingUp")):
    if (env_name.startswith("CartPoleSwingUpHarder")):
      print("cartpole_swingup_harder_started")
      from cartpole_swingup_harder import CartPoleSwingUpHarderEnv
      env = CartPoleSwingUpHarderEnv()
    else:
      print("cartpole_swingup_started")
      from cartpole_swingup import CartPoleSwingUpEnv
      env = CartPoleSwingUpEnv()
  elif (env_name.startswith("DreamCartPoleSwingUp")):
    print("dream_cartpole_swingup_started")
    from dream import DreamCartPoleSwingUpEnv
    env = DreamCartPoleSwingUpEnv()
  else:
    assert False, "invalid environment name."
  if (seed >= 0):
    env.seed(seed)
  '''
  print("environment details")
  print("env.action_space", env.action_space)
  print("high, low", env.action_space.high, env.action_space.low)
  print("environment details")
  print("env.observation_space", env.observation_space)
  print("high, low", env.observation_space.high, env.observation_space.low)
  assert False
  '''
  return env
