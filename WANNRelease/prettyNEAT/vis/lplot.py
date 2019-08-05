"""
Laconic plot functions to replace some of the matplotlibs verbosity
"""
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


# -- File I/O ------------------------------------------------------------ -- #
def lsave(data,fileName):
  np.savetxt(fileName, data, delimiter=',',fmt='%1.2e')    
    
def lload(fileName):
  return np.loadtxt(fileName, delimiter=',') 


# -- Basic Plotting ------------------------------------------------------ -- #
def lplot(*args,label=[],axis=False):
  """Plots an vector, a set of vectors, with or without an x scale
  """
  fig, ax = getAxis(axis)

  if len(args) == 1: # No xscale
    x = np.arange(np.shape(args)[1])
    y = args[0]
  if len(args) == 2: # xscale given
    x = args[0]
    y = args[1]
    
  if np.ndim(y) == 2:
    for i in range(np.shape(y)[1]):
      ax.plot(x,y[:,i],'-')
      if len(label) > 0:
        ax.legend((label))     
  else:
    ax.plot(x,y,'o-')    
    
  if axis is False:    
    return fig, ax
  else:
    return ax

def ldist(x, axis=False):
  """Plots histogram with estimated distribution
  """
  fig, ax = getAxis(axis)
  
  if isinstance(x, str):
    vals = lload(x)
  else:
    vals = x
  sns.distplot(vals.flatten(),ax=ax,bins=10)
  #sns.distplot(vals.flatten(),ax=ax,hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": "g"})
  return ax


def lquart(x,y,label=[],axis=False):
  """Plots quartiles, x is a vector, y is a matrix with same length as x
  """
  if axis is not False:
    ax = axis
    fig = ax.figure.canvas 
  else:
    fig, ax = plt.subplots()
  
  q = np.percentile(y,[25,50,75],axis=1)
  plt.plot(x,q[1,:],label=label) # median
  plt.plot(x,q[0,:],'k:',alpha=0.5)
  plt.plot(x,q[2,:],'k:',alpha=0.5)
  plt.fill_between(x,q[0,:],q[2,:],alpha=0.25)
  
  return ax
  
def getAxis(axis):
  if axis is not False:
    ax = axis
    fig = ax.figure.canvas 
  else:
    fig, ax = plt.subplots()
    
  return fig,ax
# -- --------------- -- --------------------------------------------#
  
  
