import numpy as np
import argparse
import sys
from graphviz import Digraph

from wann_src import *
from domain import *

def actNum2str(actId):
    actDict = {1:('Linear','grey'),
               2:('Unsigned Step Function','green'),
               3:('Sin','yellow'),
               4:('Gausian','orange'),
               5:('tanh','navy'),
               6:('Sigmoid unsigned','hotpink'),
               7:('Inverse','black'),
               8:('Absolute Value','red'),
               9:('Relu','skyblue'),
               10:('Cosine','purple'),
               11:('Squared','lightseagreen')}

    item = actDict.get(actId)
    name, color = item[0], item[1]
    return name, color

def main(argv):
    infile  = args.infile
    outPref = args.outPref
    hyp_default = args.default
    hyp_adjust  = args.hyperparam
    view = args.view

    # Load parameters
    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp,hyp_adjust)
    input_size = games[hyp['task']].input_size
    output_size = games[hyp['task']].output_size
    in_out_labels = games[hyp['task']].in_out_labels
    in_out_labels.insert(input_size,'bias')

    # Import individual for testing
    wVec, aVec, wKey = importNet(infile)

    dim = int(np.sqrt(np.shape(wVec)[0]))
    wVec = np.reshape(wVec,(dim,dim))
    
    # aVec:[N*1], wVec:[N**2 * 1]
    graph = Digraph(engine='dot', format="png")
    graph.attr("node", shape="circle")
    for i in range(dim):
        label, color = actNum2str(aVec[i])
        if i<= input_size:
            graph.node(str(i), in_out_labels[i],\
                       style='filled', fillcolor=color, color=color)
        elif i==dim-output_size:
            graph.node(str(i), in_out_labels[-1],\
                       style='filled', fillcolor=color, color=color)
        else:
            graph.node(str(i), label,color=color)
            
    (farray, tarray) = np.where(wVec!=0)
    for i in range(len(farray)):
        graph.edge(str(farray[i]),str(tarray[i]))

    graph.render(outPref, view=view)
    

# -- --------------------------------------------------------------------- -- #
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    ''' Parse input and launch '''
    parser = argparse.ArgumentParser(description=('Show ANNs'))
    
    parser.add_argument('-i', '--infile', type=str,\
                        help='file name for genome input', default='log/test_best.out')
    
    parser.add_argument('-o', '--outPref', type=str,\
                        help='file name prefix for result input', default='log/graph')

    parser.add_argument('-d', '--default', type=str,\
                        help='default hyperparameter file', default='p/default_wan.json')

    parser.add_argument('-p', '--hyperparam', type=str,\
                        help='hyperparameter file', default=None)
  
    parser.add_argument('-v', '--view', type=str2bool,\
                        help='Visualize graph?', default=False)

    args = parser.parse_args()
    main(args)
