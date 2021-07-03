import os
import numpy as np
import copy
from .ind import exportNet

class DataGatherer():
  ''' Data recorder for WANN algorithm'''
  def __init__(self, filename, hyp): 
    """
    Args:
      filename - (string) - path+prefix of file output destination
      hyp      - (dict)   - algorithm hyperparameters
    """
    self.filename = filename # File name path + prefix
    self.p = hyp
    
    # Initialize empty fields
    self.elite = []
    self.best = []
    self.bestFitVec = []
    self.spec_fit = []
    self.field = ['x_scale','fit_med','fit_max','fit_top','fit_peak',\
                  'node_med','conn_med',\
                  'elite','best']
                  
    self.objVals = np.array([])

    for f in self.field[:-2]:
      exec('self.' + f + ' = np.array([])')
      #e.g. self.fit_max   = np.array([]) 

    self.newBest = False

  def gatherData(self, pop, species):
    # Readability
    fitness = [ind.fitness for ind in pop]
    peakfit = [ind.fitMax for ind in pop]
    nodes = np.asarray([np.shape(ind.node)[1] for ind in pop])
    conns = np.asarray([ind.nConn for ind in pop])
    
    # --- Evaluation Scale ---------------------------------------------------
    if len(self.x_scale) == 0:
      self.x_scale = np.append(self.x_scale, len(pop))
    else:
      self.x_scale = np.append(self.x_scale, self.x_scale[-1]+len(pop))
    # ------------------------------------------------------------------------ 

    
    # --- Best Individual ----------------------------------------------------
    self.elite.append(pop[np.argmax(fitness)])
    if len(self.best) == 0:
      self.best = copy.deepcopy(self.elite)
    elif (self.elite[-1].fitness > self.best[-1].fitness):
      self.best = np.append(self.best,copy.deepcopy(self.elite[-1]))
      self.newBest = True
    else:
      self.best = np.append(self.best,copy.deepcopy(self.best[-1]))   
      self.newBest = False
    # ------------------------------------------------------------------------ 

    
    # --- Generation fit/complexity stats ------------------------------------ 
    self.node_med = np.append(self.node_med,np.median(nodes))
    self.conn_med = np.append(self.conn_med,np.median(conns))
    self.fit_med  = np.append(self.fit_med, np.median(fitness))
    self.fit_max  = np.append(self.fit_max,  self.elite[-1].fitness)
    self.fit_top  = np.append(self.fit_top,  self.best[-1].fitness)
    self.fit_peak = np.append(self.fit_peak, self.best[-1].fitMax)
    # ------------------------------------------------------------------------ 


    # --- MOO Fronts ---------------------------------------------------------
    if len(self.objVals) == 0:
      self.objVals = np.c_[fitness,peakfit,conns]
    else:
      self.objVals = np.c_[self.objVals, np.c_[fitness,peakfit,conns]]
    # ------------------------------------------------------------------------ 

  def display(self):
    return    "|---| Elite Fit: " + '{:.2f}'.format(self.fit_max[-1]) \
         + " \t|---| Best Fit:  "  + '{:.2f}'.format(self.fit_top[-1]) \
         + " \t|---| Peak Fit:  "  + '{:.2f}'.format(self.fit_peak[-1])

  def save(self, gen=(-1), saveFullPop=False):
    ''' Save data to disk '''
    filename = self.filename
    pref = 'log/' + filename

    # --- Generation fit/complexity stats ------------------------------------ 
    gStatLabel = ['x_scale',\
                  'fit_med','fit_max','fit_top','fit_peak',\
                  'node_med','conn_med']
    genStats = np.empty((len(self.x_scale),0))
    for i in range(len(gStatLabel)):
      #e.g.         self.    fit_max          [:,None]
      evalString = 'self.' + gStatLabel[i] + '[:,None]'
      genStats = np.hstack((genStats, eval(evalString)))
    lsave(pref + '_stats.out', genStats)
    # ------------------------------------------------------------------------ 


    # --- Best Individual ----------------------------------------------------
    wMat = self.best[gen].wMat
    aVec = self.best[gen].aVec
    exportNet(pref + '_best.out',wMat,aVec)
    
    if gen > 1:
      folder = 'log/' + filename + '_best/'
      if not os.path.exists(folder):
        os.makedirs(folder)
      exportNet(folder + str(gen).zfill(4) +'.out',wMat,aVec)
    # ------------------------------------------------------------------------

    # --- MOO Fronts ---------------------------------------------------------
    lsave(pref + '_objVals.out',self.objVals)
    # ------------------------------------------------------------------------

  def savePop(self,pop,filename):
    folder = 'log/' + filename + '_pop/'
    if not os.path.exists(folder):
      os.makedirs(folder)

    for i in range(len(pop)):
      exportNet(folder+'ind_'+str(i)+'.out', pop[i].wMat, pop[i].aVec)

def lsave(filename, data):
  np.savetxt(filename, data, delimiter=',',fmt='%1.2e')





