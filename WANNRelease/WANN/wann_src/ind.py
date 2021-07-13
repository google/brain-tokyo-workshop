import numpy as np
import copy
import os


# -- Individual Class ---------------------------------------------------- -- #

class Ind():
  """Individual class: genes, network, and fitness
  """ 
  def __init__(self, conn, node):
    """Intialize individual with given genes
    Args:
      conn - [5 X nUniqueGenes]
             [0,:] == Innovation Number
             [1,:] == Source
             [2,:] == Destination
             [3,:] == Weight
             [4,:] == Enabled?
      node - [3 X nUniqueGenes]
             [0,:] == Node Id
             [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
             [2,:] == Activation function (as int)
  
    Attributes:
      node    - (np_array) - node genes (see args)
      conn    - (np_array) - conn genes (see args)
      nInput  - (int)      - number of inputs
      nOutput - (int)      - number of outputs
      wMat    - (np_array) - weight matrix, one row and column for each node
                [N X N]    - rows: connection from; cols: connection to
      wVec    - (np_array) - wMat as a flattened vector
                [N**2 X 1]    
      aVec    - (np_array) - activation function of each node (as int)
                [N X 1]    
      nConn   - (int)      - number of connections
      fitness - (double)   - fitness averaged over all trials (higher better)
      X fitMax  - (double)   - best fitness over all trials (higher better)
      rank    - (int)      - rank in population (lower better)
      birth   - (int)      - generation born
      species - (int)      - ID of species
    """
    self.node    = np.copy(node)
    self.conn    = np.copy(conn)
    self.nInput  = sum(node[1,:]==1)
    self.nOutput = sum(node[1,:]==2)
    self.wMat    = []
    self.aVec    = []
    self.nConn   = []
    self.fitness = [] # Mean fitness over trials
    self.fitMax  = [] # Best fitness over trials
    self.rank    = []
    self.birth   = []
    self.species = []

  def nConns(self):
    """Returns number of active connections
    """
    return int(np.sum(self.conn[4,:]))

  def express(self):
    """Converts genes to weight matrix and activation vector
    """
    order, wMat = getNodeOrder(self.node, self.conn)
    if order is not False:
      self.wMat = wMat
      self.aVec = self.node[2,order]

      wVec = self.wMat.flatten()
      wVec[np.isnan(wVec)] = 0
      self.wVec  = wVec
      self.nConn = np.sum(wVec!=0)
      return True
    else:
      return False



# -- ANN Ordering -------------------------------------------------------- -- #

def getNodeOrder(nodeG,connG):
  """Builds connection matrix from genome through topological sorting.

  Args:
    nodeG - (np_array) - node genes
            [3 X nUniqueGenes]
            [0,:] == Node Id
            [1,:] == Type (1=input, 2=output 3=hidden 4=bias)
            [2,:] == Activation function (as int)

    connG - (np_array) - connection genes
            [5 X nUniqueGenes] 
            [0,:] == Innovation Number (unique Id)
            [1,:] == Source Node Id
            [2,:] == Destination Node Id
            [3,:] == Weight Value
            [4,:] == Enabled?  

  Returns:
    Q    - [int]      - sorted node order as indices
    wMat - (np_array) - ordered weight matrix
           [N X N]

    OR

    False, False      - if cycle is found

  Todo:
    * setdiff1d is slow, as all numbers are positive ints there is probably
      a better way to do with indexing tricks and bitstrings.
  """
  conn = np.copy(connG)
  node = np.copy(nodeG)
  nIns = len(node[0,node[1,:] == 1]) + len(node[0,node[1,:] == 4])
  nOuts = len(node[0,node[1,:] == 2])
  
  # Create connection and initial weight matrices
  conn[3,conn[4,:]==0] = np.nan # disabled but still connected
  src  = conn[1,:].astype(int)
  dest = conn[2,:].astype(int)
  
  lookup = node[0,:].astype(int)
  for i in range(len(lookup)): # Can we vectorize this?
    src[np.where(src==lookup[i])] = i
    dest[np.where(dest==lookup[i])] = i
  
  wMat = np.zeros((np.shape(node)[1],np.shape(node)[1]))
  wMat[src,dest] = conn[3,:]
  connMat = wMat[nIns+nOuts:,nIns+nOuts:]
  connMat[connMat!=0] = 1

  # Topological Sort of Hidden Nodes
  edge_in = np.sum(connMat,axis=0)
  Q = np.where(edge_in==0)[0]  # Start with nodes with no incoming connections
  for i in range(len(connMat)):
    if (len(Q) == 0) or (i >= len(Q)):
      Q = []
      return False, False # Cycle found, can't sort
    edge_out = connMat[Q[i],:]
    edge_in  = edge_in - edge_out # Remove nodes' conns from total
    nextNodes = np.setdiff1d(np.where(edge_in==0)[0], Q)
    Q = np.hstack((Q,nextNodes))

    if sum(edge_in) == 0:
      break
  
  # Add In and outs back and reorder wMat according to sort
  Q += nIns+nOuts
  Q = np.r_[lookup[:nIns], Q, lookup[nIns:nIns+nOuts]]
  wMat = wMat[np.ix_(Q,Q)]
  
  return Q, wMat

def getLayer(wMat):
  """Get layer of each node in weight matrix
  Traverse wMat by row, collecting layer of all nodes that connect to you (X).
  Your layer is max(X)+1. Input and output nodes are ignored and assigned to 
  layer 0 and max(X)+1 at the end.

  Args:
    wMat  - (np_array) - ordered weight matrix
           [N X N]

  Returns:
    layer - [int]      - layer # of each node

  Todo:
    * With very large networks this might be a performance sink -- especially, 
    given that this happen in the serial part of the algorithm. There is
    probably a more clever way to do this given the adjacency matrix.
  """
  wMat[np.isnan(wMat)] = 0  
  wMat[wMat!=0]=1
  nNode = np.shape(wMat)[0]
  layer = np.zeros((nNode))
  while (True): # Loop until sorting is stable
    prevOrder = np.copy(layer)
    for curr in range(nNode):
      srcLayer=np.zeros((nNode))
      for src in range(nNode):
        srcLayer[src] = layer[src]*wMat[src,curr]   
      layer[curr] = np.max(srcLayer)+1    
    if all(prevOrder==layer):
      break
  return layer-1


# -- ANN Activation ------------------------------------------------------ -- #

def act(weights, aVec, nInput, nOutput, inPattern):
  """Returns FFANN output given a single input pattern
  If the variable weights is a vector it is turned into a square weight matrix
  
  Allows the network to return the result of several samples at once if given 
  a matrix instead of a vector of inputs:
      Dim 0 : individual samples
      Dim 1 : dimensionality of pattern (# of inputs)

  Args:
    weights   - (np_array) - ordered weight matrix or vector
                [N X N] or [N**2]
    aVec      - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
    nInput    - (int)      - number of input nodes
    nOutput   - (int)      - number of output nodes
    inPattern - (np_array) - input activation
                [1 X nInput] or [nSamples X nInput]

  Returns:
    output    - (np_array) - output activation
                [1 X nOutput] or [nSamples X nOutput]
  """
  # Turn weight vector into weight matrix
  if np.ndim(weights) < 2:
      nNodes = int(np.sqrt(np.shape(weights)[0]))
      wMat = np.reshape(weights, (nNodes, nNodes))
  else:
      nNodes = np.shape(weights)[0]
      wMat = weights
  wMat[np.isnan(wMat)]=0

  # Vectorize input
  if np.ndim(inPattern) > 1:
      nSamples = np.shape(inPattern)[0]
  else:
      nSamples = 1

  # Run input pattern through ANN    
  nodeAct  = np.zeros((nSamples,nNodes))
  nodeAct[:,0] = 1 # Bias activation
  nodeAct[:,1:nInput+1] = inPattern

  # Propagate signal through hidden to output nodes
  iNode = nInput+1
  for iNode in range(nInput+1,nNodes):
      rawAct = np.dot(nodeAct, wMat[:,iNode]).squeeze()
      nodeAct[:,iNode] = applyAct(aVec[iNode], rawAct) 
      #print(nodeAct)
  output = nodeAct[:,-nOutput:]   
  return output

def applyAct(actId, x):
  """Returns value after an activation function is applied
  Lookup table to allow activations to be stored in numpy arrays

  case 1  -- Linear
  case 2  -- Unsigned Step Function
  case 3  -- Sin
  case 4  -- Gausian with mean 0 and sigma 1
  case 5  -- Hyperbolic Tangent [tanh] (signed)
  case 6  -- Sigmoid unsigned [1 / (1 + exp(-x))]
  case 7  -- Inverse
  case 8  -- Absolute Value
  case 9  -- Relu
  case 10 -- Cosine
  case 11 -- Squared

  Args:
    actId   - (int)   - key to look up table
    x       - (???)   - value to be input into activation
              [? X ?] - any type or dimensionality

  Returns:
    output  - (float) - value after activation is applied
              [? X ?] - same dimensionality as input
  """
  if actId == 1:   # Linear
    value = x

  if actId == 2:   # Unsigned Step Function
    value = 1.0*(x>0.0)
    #value = (np.tanh(50*x/2.0) + 1.0)/2.0

  elif actId == 3: # Sin
    value = np.sin(np.pi*x) 

  elif actId == 4: # Gaussian with mean 0 and sigma 1
    value = np.exp(-np.multiply(x, x) / 2.0)

  elif actId == 5: # Hyperbolic Tangent (signed)
    value = np.tanh(x)     

  elif actId == 6: # Sigmoid (unsigned)
    value = (np.tanh(x/2.0) + 1.0)/2.0

  elif actId == 7: # Inverse
    value = -x

  elif actId == 8: # Absolute Value
    value = abs(x)   
    
  elif actId == 9: # Relu
    value = np.maximum(0, x)   

  elif actId == 10: # Cosine
    value = np.cos(np.pi*x)

  elif actId == 11: # Squared
    value = x**2
    
  else:
    value = x

  return value

# -- Action Selection ---------------------------------------------------- -- #
def selectAct(action, actSelect):  
  """Selects action based on vector of actions

    We aren't selecting a single action:
    - Softmax: a softmax normalized distribution of values is returned
    - Default: all actions are returned 

  Args:
    action   - (np_array) - vector weighting each possible action
                [N X 1]

  Returns:
    i         - (int) or (np_array)     - chosen index
                         [N X 1]
  """    
  if actSelect == 'softmax':
    action = softmax(action)
  else:
    action = action.flatten()
  return action

def softmax(x):
  """Compute softmax values for each sets of scores in x.
  Assumes: [samples x dims]

  Args:
    x - (np_array) - unnormalized values
        [samples x dims]

  Returns:
    softmax - (np_array) - softmax normalized in dim 1
  
  Todo: Untangle all the transposes...    
  """  
  if x.ndim == 1:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
  else:
    e_x = np.exp(x.T - np.max(x,axis=1))
    return (e_x / e_x.sum(axis=0)).T



# -- File I/O ------------------------------------------------------------ -- #
""" Networks are exported as [N x (N+1] matrices, where the first NxN portion
is a weight matrix (rows==source, cols==destination) and the last column are
integers interpreted as activation functions as per the 'act' function above 
"""
def exportNet(filename,wMat, aVec):
  indMat = np.c_[wMat,aVec]
  np.savetxt(filename, indMat, delimiter=',',fmt='%1.2e')
    
def importNet(fileName):
  ind = np.loadtxt(fileName, delimiter=',')
  wMat = ind[:,:-1]     # Weight Matrix
  aVec = ind[:,-1]      # Activation functions

  # Create weight key
  wVec = wMat.flatten()
  wVec[np.isnan(wVec)]=0
  wKey = np.where(wVec!=0)[0] 

  return wVec, aVec, wKey

def vec2ind(p, wVec, aVec, wKey, rand=False):
  dim = int(np.sqrt(np.shape(wVec)[0]))
  wVec = np.reshape(wVec,(dim,dim))

  # - Create Nodes -
  nodeId = np.arange(0, len(aVec), 1)
  node = np.empty((3,len(nodeId)))
  node[0,:] = nodeId

  node[1,0] = 4 # Bias
  node[1,1:p['ann_nInput']+1] = 1 # Input
  node[1,(p['ann_nInput']+1):\
       (p['ann_nInput']+p['ann_nOutput']+1)] = 2 # Output
  node[1,(p['ann_nInput']+p['ann_nOutput']+1):] = 3 # Others

  node[2,:(p['ann_nInput']+p['ann_nOutput']+1)] = p['ann_initAct']
  node[2,(p['ann_nInput']+p['ann_nOutput']+1):] = aVec[p['ann_nInput']+p['ann_nOutput']:-1]

  def calc_index(num):
    if num < p['ann_nInput']:  # input
      return num+1
    elif num == p['ann_nInput']: # bias
      return 0
    elif num < p['ann_nInput']+p['ann_nOutput']+2: # others
      return num+p['ann_nOutput']
    else: # output
      return num-(p['ann_nOutput']+1)

  # - Create Conns -
  nConn = (p['ann_nInput']+1) * p['ann_nOutput']
  ins   = np.arange(0,p['ann_nInput']+1,1)
  outs  = (p['ann_nInput']+1) + np.arange(0,p['ann_nOutput'])

  conn = np.empty((5,nConn,))
  conn[0,:] = np.arange(0,nConn,1)
  conn[1,:] = np.tile(ins, len(outs))
  conn[2,:] = np.repeat(outs,len(ins))
  conn[3,:] = 1
  conn[4,:] = 0
  
  if rand is True:
    conn[4,:] = np.random.rand(1,nConn) < p['prob_initEnable']

  (src, dest) = np.where(wVec==1)
  j = 0
  for i in range(len(src)):
    x, y = calc_index(src[i]), calc_index(dest[i])
    if (node[1,x] == 1 or node[1,x] == 4)  and node[1,y] == 2:
      index = np.where((conn[1,:]==x) & (conn[2,:]==y))[0]
      conn[4,index] = 1
    else:
      add_conn = np.array([nConn + j, node[0,x], node[0,y], 1., 1.])
      conn = np.insert(conn, conn.shape[1], add_conn, axis=1)
      j += 1

  return conn, node
    
def importInd(path, exnum, p):
  pop = []
  if not os.path.exists(path):
      return pop

  if path.endswith('.out'):
    for i in range(exnum):
      wVec, aVec, wKey = importNet(path)
      conn, node = vec2ind(p, wVec, aVec, wKey, True)
      newInd = Ind(conn, node)
      newInd.express()
      newInd.birth = 0
      pop.append(copy.deepcopy(newInd))
    return pop

  else:
    files = os.listdir(path=path)
    if exnum > len(files):
      exnum = len(files)
    choosen = np.random.choice(files, size=exnum, replace=False)
    
    for filename in choosen:
      wVec, aVec, wKey = importNet(filename)
      conn, node = vec2ind(p, wVec, aVec, wKey)
      newInd = Ind(conn, node)
      newInd.express()
      newInd.birth = 0
      pop.append(copy.deepcopy(newInd))
    return pop
