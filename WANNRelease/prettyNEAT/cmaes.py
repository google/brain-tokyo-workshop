import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np


# NEAT and task
import cma
from neat_src import *
from domain import *

# MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Easier to read output
np.set_printoptions(precision=2, linewidth=160) 

# -- CMA-ES -------------------------------------------------------------- -- #
def master():
  # Setup environment and network
  global fileName, taskName, hyp, outPref
  task = GymTask(games[taskName], nReps=hyp['alg_nReps'])
  
  if fileName == False:
    wVec, aVec, wKey = createNet(hyp)
  else:
    wVec, aVec, wKey = importNet(fileName)

  # -- Run CMA-ES -------------------------------------------------------- -- #
  initGuess = np.random.rand(len(wKey))
  es = cma.CMAEvolutionStrategy(initGuess, hyp['initSigma'],\
    {'popsize': hyp['popSize'], 'maxiter': hyp['maxGen']} )
  saveCount = 0
  saveMod = 16
  while not es.stop():
    solutions = es.ask()              # Get new solutions
    fitness = batchMpiEval(solutions) # Calcuate Fitness
    es.tell(solutions, -fitness)      # Send fitness back to ES
    es.logger.add()                   # write data to disc to be plotted
    es.disp()

    saveCount += 1
    if (saveCount % saveMod)==0:
      wVec[wKey] = es.best.x
      matDim = int(np.sqrt(len(wVec)))
      exportNet(outPref+'best.out', wVec.reshape(matDim,matDim), aVec)


  print('*** Best fitness found:', -es.best.f)
  stopAllWorkers() 
  # -- ------------------------------------------------------------------- -- #

  # Reconstruct solution and save as weight/activation matrix
  wVec[wKey] = es.best.x
  matDim = int(np.sqrt(len(wVec)))
  exportNet(outPref+'best.out', wVec.reshape(matDim,matDim), aVec)


# -- Fixed ANN Creation--------------------------------------------------- -- #
def createNet(hyp):
    wMat = layer2mat(hyp, hyp['ann_layers'])
    aVec = np.full( (np.shape(wMat)[0]), 6) # sigmoid hidden and output
    aVec[:hyp['ann_nInput']+1] = 1 # Linear inputs

    # Create weight key
    wVec = wMat.flatten()
    wVec[np.isnan(wVec)]=0
    wKey = np.where(wVec!=0)[0]

    return wVec, aVec, wKey

def layer2mat(hyp, hLayers):
  ''' 
  Creates weight matrix of fully connected layers

  Example: layer2mat(hyp, [25,5]) # [input, [25x5] hidden, output]
  '''
  print('Layers: ',  hLayers)
  nPerLay = [1+hyp['ann_nInput']] + hLayers + [hyp['ann_nOutput']]
  nNodes = sum(nPerLay)
  adjMat = np.zeros((nNodes,nNodes))  # Adjacency Matrix  

  # Connect layers
  lastNodeId = np.cumsum(nPerLay)
  for i in range(len(lastNodeId)-1):
    # Get source and destination nodes
    if i == 0:
      src = np.arange(0,lastNodeId[i])
    else:
      src = dest
    dest= np.arange(src[-1]+1,lastNodeId[i+1])
    
    # Flag as connected in adjacency matrix
    for s in src: adjMat[s,dest] = 1 
  
  return adjMat


# -- Parallelization ----------------------------------------------------- -- #
def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    return "child"

def batchMpiEval(pop):
  ''' Sends population to workers for evaluation one batch at a time '''
  global nWorker
  nSlave = nWorker-1
  nJobs = len(pop)
  nBatch= math.ceil(nJobs/(nSlave)) # First worker is master
  seed = np.random.randint(1000)


  fitness = np.empty(nJobs, dtype=np.float64)
  i = 0 # Index of fitness we are filling
  for iBatch in range(nBatch): # Send one batch of individuals
    for iWork in range(nSlave): # (one to each worker if there)
      if i < nJobs:
        wVec = pop[i].flatten()
        nData = np.shape(wVec)[0]
        comm.send(nData, dest=(iWork)+1, tag=1)
        comm.Send(wVec,  dest=(iWork)+1, tag=2)
        comm.send(seed,  dest=(iWork)+1, tag=3)

      else:        
        nData = 0
        comm.send(nData,  dest=(iWork)+1)
      i = i+1 
  
    # Get fitness values back for that batch
    i -= nSlave
    for iWork in range(1,nSlave+1):
      if i < nJobs:
        workResult = np.empty(1, dtype='d')
        comm.Recv(workResult, source=iWork)
        fitness[i] = workResult
      i+=1
  return fitness

def slave():
  ''' Sends back fitness of any individuals sent to it '''
  # Make environment
  global fileName, taskName, hyp
  task = GymTask(games[taskName], nReps=hyp['alg_nReps'])
  
  if fileName == False:
    wVec, aVec, wKey = createNet(hyp)
  else:
    wVec, aVec, wKey = importNet(fileName)

  # Evaluate any weight vectors sent this way
  while True:
    nData = comm.recv(source=0, tag=1)  # how long is the array that's coming?
    if nData > 0:
      x = np.empty(nData, dtype='d')    # allocate space to receive weights
      comm.Recv(x, source=0, tag=2)     # recieve it
      seed = comm.recv(source=0, tag=3) # random seed as int

      # Plug values into weight matrix and evaluate
      wVec[wKey] = x
      result = task.getFitness(wVec,aVec,seed=seed)

      comm.Send(result, dest=0) # send it back
    if nData < 0: # End signal recieved
      print('Worker # ', rank, ' shutting down.')
      break

def stopAllWorkers():
  global nWorker
  # don't if parallel...
  nSlave = nWorker-1
  for iWork in range(nSlave):
    comm.send(-1, dest=(iWork)+1, tag=1)


# -- Input Parsing ------------------------------------------------------- -- #
def main(argv):
  ''' Launches optimization and evaluation scripts '''
  # Handle inputs
  global taskName, fileName, hyp, outPref
  fileName = args.filename
  outPref  = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam

  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)
  taskName = hyp['task']

  # Launch main thread and workers
  if (rank == 0):
    master()
  else:
    slave()


if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Optimize ANN with CMA-ES'))
  
 
  parser.add_argument('-i', '--filename', type=str,\
   help='file name for genome input', default=False)
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='p/default_cma.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default=None)
  
  parser.add_argument('-o', '--outPrefix', type=str,\
   help='prefix of file name for output', default='log/cma_')

  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=8)

  args = parser.parse_args()

  # Use MPI if parallel
  if "parent" == mpi_fork(args.num_worker+1): os._exit(0)

  main(args)                             
  
