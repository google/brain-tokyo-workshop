import numpy as np
import argparse
import sys
from graphviz import Digraph

from wann_src import *
from domain import *

def actNum2str(actId):
    actDict = {1:('Linear','lightslategrey'),
               2:('Unsigned Step Function','tomato'),
               3:('Sin','orange'),
               4:('Gausian','gold'),
               5:('tanh','greenyellow'),
               6:('Sigmoid unsigned','yellowgreen'),
               7:('Inverse','turquoise'),
               8:('Absolute Value','skyblue'),
               9:('Relu','dodgerblue'),
               10:('Cosine','mediumslateblue'),
               11:('Squared','mediumorchid')}

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

    in_labels = in_out_labels[:input_size] + ['bias']
    out_labels = in_out_labels[input_size:]
    
    
    # Import individual for testing
    wVec, aVec, wKey = importNet(infile)
    dim = int(np.sqrt(np.shape(wVec)[0]))
    wVec = np.reshape(wVec,(dim,dim))


    
    # aVec:[N*1], wVec:[N**2 * 1]
    graph = Digraph(engine='dot', format="svg")
    graph.attr(ranksep='1.5')
    graph.attr('graph', rankdir="LR")
    graph.attr("node", shape="oval")

    
    with graph.subgraph(name='input') as sub:
        sub.attr(rank='same')
        for i in range(input_size+1):
            sub.node(str(i), str(in_labels[i]), color='lightgrey')

    for i in range(input_size+1,dim-output_size):
        label, color = actNum2str(aVec[i])
        graph.node(str(i), label,color=color,\
            style='filled', fillcolor=color)

    with graph.subgraph(name='output') as sub:
        sub.attr(rank='same')
        for i in range(output_size):
            j = dim-output_size+i
            sub.node(str(j), str(out_labels[i]), color='black')
        
            
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
