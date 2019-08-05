import glob
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import pickle
import warnings
from matplotlib import cm

# Custom libraties
from .lplot import *


def loadPop(popFile):
  file = open(popFile,'rb')
  return pickle.load(file) 


# -- Replicates ---- ----------------------------------------------#
def viewFitFile(fname,val='Fit',label=None,axis=False):
  fig, ax = getAxis(axis)
  stats = lload(fname)
  x = stats[:,0]
  if val == 'Fit':
    fit = stats[:,[1,2,4]]
    #lplot(x,fit,label=['Min','Med', 'Top'],axis=ax)
    lplot(fit,label=['Min','Med', 'Top'],axis=ax)
  else:
    fit = stats[:,5]
    lplot(x,fit,label=['Conns'],axis=ax)
  return fig, ax
    

  
def viewReps(prefix,label=[],val='Fit', title='Fitness',\
             axis=False, getBest=False):
  fig, ax = getAxis(axis)
  fig.dpi=100
  bestRun = []    
  for pref in prefix:
    statFile = sorted(glob.glob(pref + '*stats.out'))
    if len(statFile) == 0:
      print('ERROR: No files with that prefix found (it is a list?)')
      return False
    
    for i in range(len(statFile)):
      tmp = lload(statFile[i]) 
      if i == 0:
        x = tmp[:,0]
        if val == 'Conn':
          fitVal = tmp[:,5]
        else: # Fitness
          fitVal = tmp[:,3]
          bestVal = fitVal[-1]
          bestRun.append(statFile[i])
      else:
        if np.shape(tmp)[0] != np.shape(fitVal)[0]:
          print("Incomplete file found, ignoring ", statFile[i], ' and later.')
          break
        
        if val == 'Conn':
          fitVal = np.c_[fitVal,tmp[:,5]]    
        else: # Fitness
          fitVal = np.c_[fitVal,tmp[:,3]]
          if fitVal[-1,-1] > bestVal:
            bestVal = fitVal[-1,-1]
            bestRun[-1] = statFile[i]
    
    x = np.arange(len(x))
    lquart(x,fitVal,axis=ax) # Display Quartiles

  # Legend
  if len(label) > 0:
    newLeg = []
    for i in range(len(label)):
      newLeg.append(label[i])
      newLeg.append('_nolegend_')
      newLeg.append('_nolegend_')
    warnings.filterwarnings("ignore", category=UserWarning)
    plt.gca().legend((newLeg))
  plt.title(title)
  plt.xlabel('Evaluations')
  plt.xlabel('Generations')
  if val == 'Conn':
    plt.ylabel('Median Connections')
  else: # Fitness
    plt.ylabel('Best Fitness Found')

  if getBest is True:
    return fig,ax,bestRun
  else:
    return fig,ax
# -- ------------ -- ----------------------------------------------#

def getAxis(axis):
  if axis is not False:
    ax = axis
    fig = ax.figure.canvas 
  else:
    fig, ax = plt.subplots()
    
  return fig,ax