#@title Core imports


"""
Contains the core classes for making multi-agent grid world games.
"""

import numpy as np

#import matplotlib.pyplot as plt

import tensorflow as tf
#import sonnet as snt
import six

#tfe = tf.contrib.eager

#tf.enable_eager_execution()

__metaclass__ = type

class Map:

  def __init__(self, size=20, map_config={}):
    """Maps contain data structures storing game states.  The primary data
    structures used here are the map_dict and the object_list.  map_dict
    contains dictionaries indexed by index-within-the-world.  So,
    map_dict[(i,j)] returns a dictionary of lists of Objects on the cell at
    (i,j).  Object lists are indexed within map_dict by their class name, which
    is always defined as a human-readable string, and referenced by
    Object.c_name.
    object_list is a global dictionary of lists of objects in the game.  This
    makes it easy to loop over all Agent object without having to look at every
    location on the map.  Care must be taken to update both of these data
    structures whenever transformations are applied."""

    self.size = size

    self.map_dict = {} #stores map data by index in map

    self.object_list = {} #stores lists of different types of objects, by ref

    self.fields = {} #stores any global fields that might used by different objects

    self.fields['Temperature'] = Temperature(self,
                                   'Temperature',
                                   np.array(np.ones((self.size, self.size))))

    for i in range(size):
      for j in range(size):
        self.map_dict[(i,j)] = {}


    already_placed = [] #stores indices already used
    for _, obj_data in six.iteritems(map_config):# map_config.iteritems():
      obj_fn, num_objects = obj_data

      for this_obj in range(num_objects):
        unique = False
        while not unique:

          rand_x = np.random.randint(size-2)+1 #ensures not placed on boundary
          rand_y = np.random.randint(size-2)+1

          if (rand_x, rand_y) not in already_placed:
            unique = True
            already_placed.append((rand_x,rand_y))

        obj_name = obj_fn.c_name
        #print(obj_name)


        new_obj = obj_fn(my_map = self,
                         loc = (rand_x, rand_y),
                         name=obj_name+str(this_obj))

        if obj_name not in self.map_dict[(rand_x, rand_y)]:
          self.map_dict[(rand_x, rand_y)][obj_name] = []

        if obj_name not in self.object_list:
          self.object_list[obj_name] = []

        self.map_dict[(rand_x, rand_y)][obj_name].append(new_obj)
        self.object_list[obj_name].append(new_obj)


    wall_str = Wall.c_name

    for i in range(size):
      for j in range(size):

        if i == 0 or i == size-1 or j == 0 or j == size - 1:
          if wall_str not in self.map_dict[(i,j)]:
            self.map_dict[(i,j)][wall_str] = []

          if wall_str not in self.object_list:
            self.object_list[wall_str] = []


          this_wall = Wall(my_map = self,loc = (i,j),name=wall_str+str(i)+str(j),char='#')

          self.map_dict[(i,j)][wall_str].append(this_wall)
          self.object_list[wall_str].append(this_wall)
    pass


  def add_to_map(self, obj_fn_to_add, position):
    x_pos, y_pos = position

    obj_name = obj_fn_to_add.c_name

    new_obj = obj_fn_to_add(my_map = self,
                         loc = (x_pos, y_pos),
                         name=obj_name)

    if obj_name not in self.map_dict[(x_pos, y_pos)]:
      self.map_dict[(x_pos, y_pos)][obj_name] = []

    if obj_name not in self.object_list:
      self.object_list[obj_name] = []

    self.map_dict[(x_pos, y_pos)][obj_name].append(new_obj)
    self.object_list[obj_name].append(new_obj)



  def return_objects_at(self, loc):
    return self.map_dict[(i,j)]


  def return_obj_mask(self, obj_c_name):
    """returns a 2d np array containing every occurrence of a particular object
    """
    if obj_c_name not in self.object_list:
      return np.zeros((self.size, self.size))
    else:
      base = np.zeros((self.size, self.size))
      for o in self.object_list[obj_c_name]:
        x,y = o.loc
        base[x,y]=1
      return base

  def pretty_print(self):

    all_rows = []
    #row_str = []
    for i in range(self.size):
      row_str = []
      for j in range(self.size):
        keys = self.map_dict[(i,j)].keys()
        wrote_key = False
        if keys != []: #empty if no Objects at cell (i,j)
          for k in keys:
            if self.map_dict[(i,j)][k]!=[] and wrote_key == False:
              row_str.append(self.map_dict[(i,j)][k][0].char)
              wrote_key = True

        if wrote_key == False:
          row_str.append(' ')
      all_rows.append(row_str)


    pretty_out = '\n'.join([''.join(r) for r in all_rows])
    return pretty_out

#Super class for objects in the gridworld that occupy single spaces
class Object(object):
  c_name = 'Object'

  def __init__(self, my_map, loc, name, char='O'):
    """Super class for anything that can be put into the map.  Exists so that
    it's easy to reason about the shared methods of everything on a map."""
    self.loc = loc
    self.parent_map = my_map
    self.name = name
    self.char = char

#Super class for potentially map-spanning effects.
class Field(object):
  c_name = 'Field'

  def __init__(self, my_map, name, initial_field):
    """Super class for environmental effects that are like fields--e.g. temp,
    water-level, poisonous gas cloud, etc.  At the lowest level, these are float
    valued numpy arrays that store the field data.  At a design level, fields
    are mediators of interactions between Object and Agent subclasses.  For
    example, Fire increases the local temperature field.  High temperature harms
    agents--but also bakes apples."""
    self.parent_map = my_map
    self.name = name
    self.field = initial_field

class Temperature(Field):
  c_name = 'Temperature'

  def __init__(self, my_map, name, initial_field):
    super(Temperature, self).__init__(my_map=my_map,
                                      name=name,
                                      initial_field = initial_field)
    self.ceiling = 10*np.ones((len(self.field),len(self.field)))
    self.radiative = .1

  def update(self):
    """Performs a discrete heat equation like update to the full temperature
    field, with some assumptions about radiative loss to the sky.  The modeling
    assumption here is that the sky is a fixed reservoir at constant T=10, and
    that any other fixed temperature reservoirs will update the temperature
    value at their location within their own update functions."""


    left = np.roll(self.field, (1,0), axis=(0,1))
    right = np.roll(self.field, (-1,0), axis=(0,1))
    up = np.roll(self.field, (0,1), axis=(0,1))
    down = np.roll(self.field, (0,-1), axis=(0,1))

    #ur = np.roll(self.field, (-1,1), axis=(0,1))
    #ul = np.roll(self.field, (1,1), axis=(0,1))
    #dr = np.roll(self.field, (-1,-1), axis=(0,1))
    #dl = np.roll(self.field, (1,-1), axis=(0,1))

    """This equation is a single iteration of a discretized form of newton's
    law of cooling with a radiative correction."""
    self.field = self.field + .1*((self.ceiling + left + right +\
                                   up + down #+ ur + ul + dr + dl \
                                   - 5*self.field) -\
                                  self.radiative*self.field)


class Environmental(Object):
  c_name='Environmental'

  def __init__(self, my_map, loc, name, char='E'):
    """Parent class of anything that isn't an agent.  Subclasses are things like
    walls, apples, fire, etc.  Aiming for generality, here, because the more
    similar environmental objects are, structurally, the easier it is to allow
    interactions.  And interactions are FUN."""
    super(Environmental, self).__init__(my_map=my_map,
                                        loc=loc,
                                        name=name,
                                        char= char)

    self.traversable = None
    self.edibible = None
    self.held = False #True if being held by an Agent
    self.consume_value = 0.0

    pass

  def update(self):
    """Catchall method for any interesting mechanics that this particular
    environmental object might perform upon the world (i.e., fire hurts agents
    so could be implemented here.)"""

    pass



class Action(object):

  def __init__(self, my_agent=None, my_map=None):
    """The base class for Actions.  In general, Actions are maps from
    map dictionaries to map dictionaries.  Engine logic is as
    follows:
    1. Agents all perceive world simultaneously with agent.observe() calls.
    2. Agents propose actions to action stack.
    3. Action stack is traversed randomly (modify this later?) and attempts
    each action.  Each action def must take care to set validate_action() method
    to see if an action is really possible."""
    self.parent_map = my_map
    self.agent = my_agent


  def validate_action(self):
    """Returns a boolean for whether this action is still possible.  Must be
    implemented on case-by-case basis for each possible action in the world."""
    pass

  def act(self):
    """Implements a transformation to the environment.  Examples include:
    movement: Agent moves cells.
    pick up/drop item: Agent acquires an item
    consume item: Agent consumes an item in the world."""
    pass

class Move(Action):

  def __init__(self, my_agent, my_map, new_loc):
    super(Move,self).__init__(my_agent=my_agent, my_map=my_map)

    self.new_loc = new_loc

  def validate_action(self):
    if 'Wall' not in self.parent_map.map_dict[self.new_loc]:
      return True
    else:
      return False

  def act(self):
    """Actually implements the action if the validate_action call passes,
    else, does nothing."""
    if self.validate_action():
      this_agent = self.agent

      old_loc = this_agent.loc
      new_loc = self.new_loc
      c_name = this_agent.c_name
      this_map_dict = self.parent_map.map_dict

      #remove agent ref from old map dictionary
      this_map_dict[old_loc][c_name] = [k for k in this_map_dict[old_loc][c_name]
                                   if k!=this_agent]
      #Update agent location
      this_agent.loc=(new_loc)

      #add agent ref to new map location
      if c_name not in this_map_dict[new_loc]: #make dict if it doesn't exist
        this_map_dict[new_loc][c_name] = []
      this_map_dict[new_loc][c_name].append(this_agent)

class PickDrop(Action):
  """Action definition for picking up/dropping an Environmental instance.
  Executing this action on top of a cell containing an environmental picks it
  up.  Executing it again drops the Environmental."""

  def __init__(self, my_agent, my_map, obj_to_pickdrop):
    super(PickDrop,self).__init__(my_agent=my_agent, my_map=my_map)

    self.obj_to_pickdrop = obj_to_pickdrop

  def validate_action(self):
    obj_to_pickdrop = self.obj_to_pickdrop

    if (not obj_to_pickdrop.held and self.agent.inventory == []) \
    or (obj_to_pickdrop.held and obj_to_pickdrop in self.agent.inventory):
      return True
    else:
      return False

  def act(self):
    """Actually implements the pickdrop if the validate_action call passes,
    else, does nothing."""


    if self.validate_action():
      this_agent = self.agent
      cur_loc = this_agent.loc
      c_name = this_agent.c_name
      this_map_dict = self.parent_map.map_dict

      obj_to_pickdrop = self.obj_to_pickdrop
      obj_c_name = obj_to_pickdrop.c_name

      #if object is not being held, pick it up
      if not obj_to_pickdrop.held:

        obj_to_pickdrop.held = True
        this_map_dict[cur_loc][obj_c_name] = \
        [k for k in this_map_dict[cur_loc][obj_c_name] if k!=obj_to_pickdrop]

        this_agent.inventory.append(obj_to_pickdrop)

      #, otherwise, put it down
      else:
        this_agent.inventory = []
        if obj_c_name not in this_map_dict[cur_loc]:
          this_map_dict[cur_loc][obj_c_name] = []

        obj_to_pickdrop.held = False
        this_map_dict[cur_loc][obj_c_name].append(obj_to_pickdrop)


class Eat(Action):
  """Action definition for consuming an Environmental.  Executing this action
  consumes whichever item is currently in the 0 slot of the agent's inventory"""

  def __init__(self, my_agent, my_map):
    super(Eat,self).__init__(my_agent=my_agent, my_map=my_map)

  def validate_action(self):

    if (self.agent.inventory != []):
      return True
    else:
      return False

  def act(self):
    """Actually implements the Eat if the validate_action call passes,
    else, does nothing."""

    if self.validate_action():
      item_to_eat = self.agent.inventory.pop(0)
      item_to_eat.held = False
      self.agent.happiness += item_to_eat.consume_value
      self.agent.parent_map.object_list[item_to_eat.c_name] = \
      [o for o in self.agent.parent_map.object_list[item_to_eat.c_name] \
       if o != item_to_eat]


class Agent(Object):
  c_name = 'Agent'

  def __init__(self, my_map, loc, name, char='A'):
    """The base class for critters in the world.  Subclass needs to implement
    at least all of these methods"""
    super(Agent, self).__init__(my_map=my_map,
                                loc=loc,
                                name=name,
                                char= char)
    self.inventory = [] #Can hold one Environmental object
    self.happiness = 0.0

    pass

  def observe(self):
    """Returns a list of perceptions available to the agent."""
    pass

  def propose_action(self):
    """Returns an instance of something that inherits from the Action class.
    Actions aren't necessarily successful. Two critters can't eat the same
    fruit, or pick up the same fruit, for example."""
    pass


class RightMover(Agent):
  c_name = 'RightMover'

  def __init__(self, my_map, loc, name, char='R'):
    """A simple agent that always moves right.  For testing movement logic."""
    super(RightMover, self).__init__(my_map=my_map,
                                     loc=loc,
                                     name=name,
                                     char= char)

  def propose_action(self):
    loc_x, loc_y = self.loc
    this_action = Move(my_agent=self, my_map=self.parent_map, new_loc=(loc_x,loc_y+1))
    #print(this_action)
    return this_action

class PickerWalker(Agent):
  c_name = 'PickerWalker'
  rand_dir = {0: [1,0], 1: [0,1], 2: [-1,0], 3: [0,-1]}

  def __init__(self, my_map, loc, name, char='P'):
    """A simple agent that randomly moves and tries to pick up apples if it
    can.  Should deplete apples on ground fairly quickly."""
    super(PickerWalker, self).__init__(my_map=my_map,
                                     loc=loc,
                                     name=name,
                                     char= char)

  def propose_action(self):

    picked = False

    if 'Apple' in self.parent_map.map_dict[self.loc]:
      if self.parent_map.map_dict[self.loc]['Apple'] != []:
        apple_to_pick = self.parent_map.map_dict[self.loc]['Apple'][0]

        this_action = PickDrop(self, self.parent_map, apple_to_pick)

        picked = True
        return this_action

    if picked == False:
      loc_x, loc_y = self.loc

      r = np.random.randint(4)

      new_dir = (PickerWalker.rand_dir[r][0] + loc_x,
                 PickerWalker.rand_dir[r][1]+loc_y)

      this_action = Move(my_agent=self, my_map=self.parent_map, new_loc=new_dir)
      #print(this_action)
      return this_action

class LookerPickerWalker(Agent):
  c_name = 'LookerPickerWalker'
  rand_dir = {0: [1,0], 1: [0,1], 2: [-1,0], 3: [0,-1]}

  def __init__(self, my_map, loc, name, char='P', vision_module=None):
    """A simple agent that randomly moves and tries to pick up apples if it
    can.  Can also 'see' nearby apples, and can use this information to choose
    which direction to move. Should deplete apples on ground fairly quickly
    if vision model is good.
    args:
    vision_module: a sonnet module that takes 9x9 numpy arrays as input, and
    outputs a 1x4 np array"""
    super(LookerPickerWalker, self).__init__(my_map=my_map,
                                     loc=loc,
                                     name=name,
                                     char= char)

    #self.vision = snt.Linear(4)
    #must call snt module once to get it to fix variable shapes
    #self.vision(tf.constant(1.0, dtype=tf.float32, shape=(1,81)))

  def padded_slice(img, sl):
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
    cur_x, cur_y = self.loc
    min_x = max(0, cur_x-4)
    max_x = min(self.parent_map.size, cur_x+5)

    min_y = max(0, cur_y-4)
    max_y = min(self.parent_map.size, cur_y+5)
    #get mask of apple locations and logical AND it with this bound
    v_mask = np.zeros((self.parent_map.size, self.parent_map.size))

    for i in range(9):
      for j in range(9):
        this_x = cur_x-4+i
        this_y = cur_y-4+j
        if this_x >= min_x and this_x <= max_x and\
           this_y >= min_y and this_y <= max_y:
          v_mask[this_x, this_y] = 1
    print(v_mask)

    visual_window = my_game.world_map.return_obj_mask('Apple') * v_mask

    return padded_slice(visual_window, [cur_x-4, cur_x+5, cur_y-4, cur_y+5])

  def propose_action(self):

    #if Apple in inventory, eat it.
    if self.inventory != []:
      this_action = Eat(self, self.parent_map)
      return this_action

    #else, if Apple on ground, pick it up
    picked = False

    if 'Apple' in self.parent_map.map_dict[self.loc]:
      if self.parent_map.map_dict[self.loc]['Apple'] != []:
        apple_to_pick = self.parent_map.map_dict[self.loc]['Apple'][0]

        this_action = PickDrop(self, self.parent_map, apple_to_pick)

        picked = True
        return this_action


    #else, randomly wiggle around looking for apples
    if picked == False:
      loc_x, loc_y = self.loc

      r = np.random.randint(4)

      new_dir = (LookerPickerWalker.rand_dir[r][0] + loc_x,
                 LookerPickerWalker.rand_dir[r][1]+loc_y)

      this_action = Move(my_agent=self, my_map=self.parent_map, new_loc=new_dir)
      #print(this_action)
      return this_action

class ControllableWalker(LookerPickerWalker):
  c_name = 'ControllableWalker'
  rand_dir = {0: [1,0], 1: [0,1], 2: [-1,0], 3: [0,-1]}

  move_map = {'w': [1,0], 's': [-1,0], 'a':[0,1], 'd':[0,-1],
          0: [1,0], 1: [-1,0], 2: [0,1], 3: [0,-1], 4: [0, 0], 5: True}

  def __init__(self, my_map, loc, name, char='C', vision_module=None):
    """A simple agent that randomly moves and tries to pick up apples if it
    can.  Can also 'see' nearby apples, and can use this information to choose
    which direction to move. Should deplete apples on ground fairly quickly
    if vision model is good.
    args:
    vision_module: a sonnet module that takes 9x9 numpy arrays as input, and
    outputs a 1x4 np array"""
    super(ControllableWalker, self).__init__(my_map=my_map,
                                     loc=loc,
                                     name=name,
                                     char= char)

    #self.vision = snt.Linear(4)
    #must call snt module once to get it to fix variable shapes
    #self.vision(tf.constant(1.0, dtype=tf.float32, shape=(1,81)))

  def propose_action(self, user_action = None):

    if user_action != None:
      loc_x, loc_y = self.loc
      act = ControllableWalker.move_map[user_action]

      new_dir = (act[0] + loc_x, act[1]+loc_y)

      this_action = Move(my_agent=self, my_map=self.parent_map, new_loc=new_dir)
      #print(this_action)
      return this_action

    else:
      #if Apple in inventory, eat it.
      if self.inventory != []:
        this_action = Eat(self, self.parent_map)
        return this_action

      #else, if Apple on ground, pick it up
      picked = False

      if 'Apple' in self.parent_map.map_dict[self.loc]:
        if self.parent_map.map_dict[self.loc]['Apple'] != []:
          apple_to_pick = self.parent_map.map_dict[self.loc]['Apple'][0]

          this_action = PickDrop(self, self.parent_map, apple_to_pick)

          picked = True
          return this_action


      #else, randomly wiggle around looking for apples
      if picked == False:
        loc_x, loc_y = self.loc

        r = np.random.randint(4)

        new_dir = (ControllableWalker.rand_dir[r][0] + loc_x,
                   ControllableWalker.rand_dir[r][1]+loc_y)

        this_action = Move(my_agent=self, my_map=self.parent_map, new_loc=new_dir)
        #print(this_action)
        return this_action


#Mapgen:

class Wall(Environmental):
  c_name = 'Wall'

  def __init__(self, my_map, loc, name, char='W'):
    super(Wall, self).__init__(my_map=my_map, loc=loc, name=name, char=char)

    self.traversable = False
    self.edible = False

class Fire(Environmental):
  c_name = 'Fire'

  def __init__(self, my_map, loc, name, char='F'):
    super(Fire, self).__init__(my_map=my_map, loc=loc, name=name, char=char)
    self.traversable = True
    self.edible = True #consuming Fire kills the agent
    self.consume_value = -10.0

  def update(self):
    assert 'Temperature' in self.parent_map.fields

    #TODO: This is a pretty nasty side-effect.  Would be nice if temperature
    #didn't depend on the existence of an object in the parent class.
    self.parent_map.fields['Temperature'].field[self.loc] = 2000.0

class Apple(Environmental):
  c_name='Apple'

  def __init__(self, my_map, loc, name, char='a'):
    super(Apple, self).__init__(my_map=my_map, loc=loc, name=name, char=char)
    self.traversable = True
    self.edible = True
    self.consume_value = 1.0

class Stick(Environmental):
  c_name='Stick'

  def __init__(self, my_map, loc, name, char='s'):
    super(Stick, self).__init__(my_map=my_map, loc=loc, name=name, char=char)
    self.traversable = True
    self.edible = False
    self.consume_value = 0.0 #if you somehow consume this, no consumption value
