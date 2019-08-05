import numpy as np
import time
import sys
import random

from domain.make_env import make_env
from .ind import *


class Task():
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
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect

    if not paramOnly:
      self.env = make_env(game.env_name)

    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))    
  

  def testInd(self, wVec, aVec, view=False,seed=-1):
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
      return reward
    else:
      totalReward = reward
    
    for tStep in range(self.maxEpisodeLength): 
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
      action = selectAct(annOut,self.actSelect) 
      state, reward, done, info = self.env.step(action)
      totalReward += reward  
      if view:
        #time.sleep(0.01)
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      if done:
        break

    return totalReward

# -- 'Weight Agnostic Network' evaluation -------------------------------- -- #
  def setWeights(self, wVec, wVal):
    """Set single shared weight of network
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      wVal    - (float)    - value to assign to all weights
  
    Returns:
      wMat    - (np_array) - weight matrix with single shared weight
                [N X N]
    """
    # Create connection matrix
    wVec[np.isnan(wVec)] = 0
    dim = int(np.sqrt(np.shape(wVec)[0]))    
    cMat = np.reshape(wVec,(dim,dim))
    cMat[cMat!=0] = 1.0

    # Assign value to all weights
    wMat = np.copy(cMat) * wVal 
    return wMat


  def getDistFitness(self, wVec, aVec, hyp, \
                    seed=-1,nRep=False,nVals=6,view=False,returnVals=False):
    """Get fitness of a single individual with distribution of weights
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
      hyp     - (dict)     - hyperparameters
        ['alg_wDist']        - weight distribution  [standard;fixed;linspace]
        ['alg_absWCap']      - absolute value of highest weight for linspace
  
    Optional:
      seed    - (int)      - starting random seed for trials
      nReps   - (int)      - number of trials to get average fitness
      nVals   - (int)      - number of weight values to test

  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = hyp['alg_nReps']

    # Set weight values to test WANN with
    if (hyp['alg_wDist'] == "standard") and nVals==6: # Double, constant, and half signal 
      wVals = np.array((-2,-1.0,-0.5,0.5,1.0,2))
    else:
      wVals = np.linspace(-self.absWCap, self.absWCap ,nVals)


    # Get reward from 'reps' rollouts -- test population on same seeds
    reward = np.empty((nRep,nVals))
    for iRep in range(nRep):
      for iVal in range(nVals):
        wMat = self.setWeights(wVec,wVals[iVal])
        if seed == -1:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=seed,view=view)
        else:
          reward[iRep,iVal] = self.testInd(wMat, aVec, seed=seed+iRep,view=view)
          
    if returnVals is True:
      return np.mean(reward,axis=0), wVals
    return np.mean(reward,axis=0)
 
