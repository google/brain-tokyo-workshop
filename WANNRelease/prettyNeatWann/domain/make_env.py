import numpy as np
import gym
from matplotlib.pyplot import imread


def make_env(env_name, seed=-1, render_mode=False):

  # -- Bipedal Walker ------------------------------------------------ -- #
  if (env_name.startswith("BipedalWalker")):
    if (env_name.startswith("BipedalWalkerHardcore")):
      import Box2D
      from domain.bipedal_walker import BipedalWalkerHardcore
      env = BipedalWalkerHardcore()
    elif (env_name.startswith("BipedalWalkerMedium")): 
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()
      env.accel = 3
    else:
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()


  # -- VAE Racing ---------------------------------------------------- -- #
  elif (env_name.startswith("VAERacing")):
    from domain.vae_racing import VAERacing
    env = VAERacing()
    
  # -- Classification ------------------------------------------------ -- #
  elif (env_name.startswith("Classify")):
    from domain.classify_gym import ClassifyEnv
    if env_name.endswith("digits"):
      from domain.classify_gym import digit_raw
      trainSet, target  = digit_raw()
    
    if env_name.endswith("mnist784"):
      from domain.classify_gym import mnist_784
      trainSet, target  = mnist_784()
    
    if env_name.endswith("mnist256"):
      from domain.classify_gym import mnist_256
      trainSet, target  = mnist_256()

    env = ClassifyEnv(trainSet,target)  


  # -- Cart Pole Swing up -------------------------------------------- -- #
  elif (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
    if (env_name.startswith("CartPoleSwingUp_Hard")):
      env.dt = 0.01
      env.t_limit = 200




  # -- Other  -------------------------------------------------------- -- #
  else:
    env = gym.make(env_name)

  if (seed >= 0):
    domain.seed(seed)

  return env