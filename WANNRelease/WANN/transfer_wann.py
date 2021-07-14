import argparse
import numpy as np

from wann_src import loadHyp, updateHyp
from wann_src.ind import *
from domain import *

def transferNet(from_size, to_size, prob, wVec, aVec, wKey):
    dim = int(np.sqrt(np.shape(wVec)[0]))
    wVec = np.reshape(wVec,(dim,dim))
    
    other_node = aVec[from_size[0]+1:-from_size[1]]
    to_dim = to_size[0]+1+len(other_node)+to_size[1]
    
    to_aVec = np.full(to_dim, 1.)
    to_aVec[to_size[0]+1:-to_size[1]] = other_node

    other_conn = wVec[from_size[0]+1:-from_size[1],from_size[0]+1:-from_size[1]]
    to_wVec = np.zeros((to_dim,to_dim))
    to_wVec = np.random.rand(to_dim,to_dim) < prob
    to_wVec = to_wVec.astype(float)
    to_wVec[to_size[0]+1:-to_size[1],to_size[0]+1:-to_size[1]] = other_conn

    return to_wVec, to_aVec, wKey

def main(args):
    hyp_default = args.default
    p = loadHyp(pFileName=hyp_default)
    
    updateHyp(p,args.inHyperparam)
    from_size = (p['ann_nInput'], p['ann_nOutput'])
    updateHyp(p,args.outHyperparam)
    to_size = (p['ann_nInput'], p['ann_nOutput'])
    prob = p['prob_initEnable']

    wVec, aVec, wKey = importNet(args.inFilename)
    wMat, aVec, wKey = transferNet(from_size, to_size, prob,wVec, aVec, wKey)

    exportNet(args.outFilename, wMat, aVec)
    
if __name__ == "__main__":
  ''' Parse input and launch '''
  # python transfer_wann.py -ip p/swing_8gen.json -op p/biped.json -if log/import8*2_best.out
  
  parser = argparse.ArgumentParser(description=('Change the number of inputs / outputs of the existing network'))

  parser.add_argument('-d', '--default', type=str,\
                      help='default hyperparameter file', default='p/default_wan.json')

  parser.add_argument('-ip', '--inHyperparam', type=str,\
                      help='hyperparameter file for input', default='p/laptop_swing.json')

  parser.add_argument('-op', '--outHyperparam', type=str,\
                      help='hyperparameter file', default='p/laptop_swing.json')

  parser.add_argument('-if', '--inFilename', type=str,\
                      help='file name for input', default='log/test_best.out')

  parser.add_argument('-of', '--outFilename', type=str,\
                      help='file name for output', default='log/transnet.out')
  

  args = parser.parse_args()

  main(args)
