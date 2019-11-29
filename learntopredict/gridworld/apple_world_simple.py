import os
import gym
import sys
from gym import spaces
import numpy as np
import tensorflow as tf
from core import *


class appleworld_gym_simple(gym.Env):
  def __init__(self, size=12):
    """Initializes a 7x7 gridworld.
    Reward +1 for navigating to the apple.
    
    
    Action space is:
    0: Up
    1: Down
    2: Left
    3: Right
    """
    
    self.size = size
    
    self.world_map = Map(size=self.size, map_config = {'agents': (ControllableWalker, 1), 'apples': (Apple, 30), 'fires': (Fire, 30)})
      
    self.action_stack = []
    
    self.reward_range = (0,10)
    self.action_space = gym.spaces.Discrete(5)
    
    self.global_step = 0
    self.total_reward = 0
    self.step_on_fire = 0

    

    self.extra_steps = min(100, 2*int(1./(np.random.rand())))

    self.reset()

  def update_world(self, action=None):
    if action!=None:
      for a in self.world_map.object_list['ControllableWalker']:
        #print(a.propose_action())
        self.action_stack.append(a.propose_action(user_action=action))

    for a in self.action_stack:
      #print(a)
      a.act()
    self.action_stack[:] = []



  def step(self, action):
    """Run one timestep of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.
    Accepts an action and returns a tuple (observation, reward, done, info).
    Args:
        action (object): an action provided by the environment
    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """
    self.update_world(action=action)
    obs_apples = self.world_map.return_obj_mask('Apple')
    obs_player = self.world_map.return_obj_mask('ControllableWalker')
    obs_fire_full = self.world_map.return_obj_mask('Fire')
    #obs_wall = self.world_map.return_obj_mask('Wall')
    #obs = np.stack([obs_apples, obs_player, obs_fire, obs_wall],axis=0)
    
    obs_ap, obs_fire = self.observe()
    
    obs = np.stack([obs_ap, obs_fire], axis=0)
    
    reward = 0
    done = False
    
    self.global_step+=1
    
    if np.sum(np.abs(np.subtract(obs_apples, obs_player)))==np.sum(obs_apples)-1:
      reward += 6

      #print("Found apple!")
      for a in self.world_map.object_list['ControllableWalker']:
        #ControllableWalkers will pick up any apple they're standing ong
        #ControllableWalkers will also eat any Apple they're holding
        #The below pair of calls ensures the agent consumes the apple this turn.
        a.propose_action().act() #picks up apple
        a.propose_action().act() #eats apple
      
      
    if np.sum(np.abs(np.subtract(obs_fire_full, obs_player)))==np.sum(obs_fire_full)-1:
      reward -= 8
      self.step_on_fire+=1
      #print('ON FIRE!') 
    #if reward == 1:
    #  done = True
    #else:
    #  done = False
    if self.global_step >= 6 + self.extra_steps:
      done=True
      #print("episode over!")
      #print(self.total_reward)

    reward += 1 #+1 reward for surviving this step

    self.total_reward += reward

    #if self.total_reward < 0:
    #  #print("done!")
    #  done = True
    
    #if self.step_on_fire >= 2:
    #  done = True

    if done:
        self.total_reward = self.total_reward / (6 + self.extra_steps)
        return obs, self.total_reward, done, None

    #return obs, reward, done, None
    return obs, 0, done, None


  def reset(self):
    """Resets the state of the environment and returns an initial observation.
    Returns: observation (object): the initial observation of the
        space.
    """

    self.extra_steps = min(100, 2*int(1./(np.random.rand())))

    self.world_map = Map(size = self.size, map_config = {'agents': (ControllableWalker, 1), 'apples': (Apple, 30), 'fires': (Fire, 30)})
    self.global_step = 0
    self.total_reward = 0
    self.step_on_fire = 0
    #obs_apples = self.world_map.return_obj_mask('Apple')
    #bs_player = self.world_map.return_obj_mask('ControllableWalker')
    #obs_fire = self.world_map.return_obj_mask('Fire')
    #obs_wall = self.world_map.return_obj_mask('Wall')
    #obs = np.stack([obs_apples, obs_player, obs_fire, obs_wall],axis=0)
    
    obs_ap, obs_fire = self.observe()
    
    obs = np.stack([obs_ap, obs_fire], axis=0)

    return obs

  def render(self, mode='human'):
    """Renders the environment.
    The set of supported modes varies per environment. (And some
    environments do not support rendering at all.) By convention,
    if mode is:
    - human: render to the current display or terminal and
      return nothing. Usually for human consumption.
    - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
      representing RGB values for an x-by-y pixel image, suitable
      for turning into a video.
    - ansi: Return a string (str) or StringIO.StringIO containing a
      terminal-style text representation. The text can include newlines
      and ANSI escape sequences (e.g. for colors).
    Note:
        Make sure that your class's metadata 'render.modes' key includes
          the list of supported modes. It's recommended to call super()
          in implementations to use the functionality of this method.
    Args:
        mode (str): the mode to render with
        close (bool): close all open renderings
    Example:
    class MyEnv(Env):
        metadata = {'render.modes': ['human', 'rgb_array']}
        def render(self, mode='human'):
            if mode == 'rgb_array':
                return np.array(...) # return RGB frame suitable for video
            elif mode is 'human':
                ... # pop up a window and render
            else:
                super(MyEnv, self).render(mode=mode) # just raise an exception
    """
    print("Player loc: ", self.world_map.object_list['ControllableWalker'][0].loc)
    num_apples = np.sum(self.world_map.return_obj_mask('Apple'))
    if num_apples > 0:
        print("Apple loc: ", self.world_map.object_list['Apple'][0].loc)
    else:
        print("Apple eaten!")

  def close(self):
    """Override _close in your subclass to perform any necessary cleanup.
    Environments will automatically close() themselves when
    garbage collected or when the program exits.
    """
    
    pass

  def seed(self, seed=None):
    """Sets the seed for this env's random number generator(s).
    Note:
        Some environments use multiple pseudorandom number generators.
        We want to capture all such seeds used in order to ensure that
        there aren't accidental correlations between multiple generators.
    Returns:
        list<bigint>: Returns the list of seeds used in this env's random
          number generators. The first value in the list should be the
          "main" seed, or the value which a reproducer should pass to
          'seed'. Often, the main seed equals the provided 'seed', but
          this won't be true if seed=None, for example.
    """
    #logger.warn("Could not seed environment %s", self)
    return

  @property
  def unwrapped(self):
    """Completely unwrap this env.
    Returns:
        gym.Env: The base non-wrapped gym.Env instance
    """
    return self

  def __str__(self):
    if self.spec is None:
        return '<{} instance>'.format(type(self).__name__)
    else:
        return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

  def padded_slice(self, img, sl):
    """Needed a padded slice definition so we can easily take logical ANDs of
    masks containing locations of apples and the visual window of the agent."""
    output_shape = np.asarray(np.shape(img))
    output_shape[0] = sl[1] - sl[0]
    output_shape[1] = sl[3] - sl[2]
    src = [max(sl[0], 0),
           min(sl[1], img.shape[0]),
           max(sl[2], 0),
           min(sl[3], img.shape[1])]
    dst = [src[0] - sl[0], src[1] - sl[0],
           src[2] - sl[2], src[3] - sl[2]]
    output = np.zeros(output_shape)
    output[dst[0]:dst[1],dst[2]:dst[3]] = img[src[0]:src[1],src[2]:src[3]]
    return output

  def observe(self):
    #set observation bounds so we don't look outside of the map
    cur_x, cur_y = self.world_map.object_list['ControllableWalker'][0].loc
    min_x = max(0, cur_x-4)
    max_x = min(self.world_map.size, cur_x+5)

    min_y = max(0, cur_y-4)
    max_y = min(self.world_map.size, cur_y+5)
    #get mask of apple locations and logical AND it with this bound
    v_mask = np.zeros((self.world_map.size, self.world_map.size))

    for i in range(5):
      for j in range(5):
        this_x = cur_x-2+i
        this_y = cur_y-2+j
        if this_x >= min_x and this_x < max_x and\
           this_y >= min_y and this_y < max_y:
          v_mask[this_x, this_y] = 1
    #print(v_mask)

    visual_window = self.world_map.return_obj_mask('Apple') * v_mask

    fire_window = self.world_map.return_obj_mask('Fire') * v_mask
    
    #print(visual_window, [cur_x-2, cur_x+3, cur_y-2, cur_y+3])
    
    visual_window = self.padded_slice(visual_window, [cur_x-2, cur_x+3, cur_y-2, cur_y+3])
    fire_window = self.padded_slice(fire_window, [cur_x-2, cur_x+3, cur_y-2, cur_y+3])
    
    return visual_window, fire_window
