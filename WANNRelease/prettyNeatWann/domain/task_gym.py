import random
import numpy as np
import sys
from domain.make_env import make_env
from neat_src import *


class GymTask():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1): 
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """
    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers      
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]
  
    # Environment
    self.nReps = nReps
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    if not paramOnly:
      self.env = make_env(game.env_name)
    
    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))    
  
  def getFitness(self, wVec, aVec, view=False, nRep=False, seed=-1):
    """Get fitness of a single individual.
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      nReps   - (nReps)    - number of trials to get average fitness
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = self.nReps
    wVec[np.isnan(wVec)] = 0
    reward = np.empty(nRep)
    for iRep in range(nRep):
      if seed > 0:
        seed = seed+iRep
      reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed)
    fitness = np.mean(reward)
    return fitness

  def testInd(self, wVec, aVec, hyp=None, view=False,seed=-1):
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    if seed >= 0:
      random.seed(seed)
      np.random.seed(seed)
      self.env.seed(seed)

    state = self.env.reset()
    self.env.t = 0
    annOut = act(wVec, aVec, self.nInput, self.nOutput, state)  
    action = selectAct(annOut,self.actSelect)    
    state, reward, done, info = self.env.step(action)
    
    if self.maxEpisodeLength == 0:
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      return reward
    else:
      totalReward = reward
    
    for tStep in range(self.maxEpisodeLength): 
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
      action = selectAct(annOut,self.actSelect) 
      state, reward, done, info = self.env.step(action)
      totalReward += reward  
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      if done:
        break
    return totalReward
