import numpy as np
import argparse
import sys
import networkx as nx
import matplotlib.pyplot as plt

from wann_src import *
from domain import *

def DOT2list(infile):
    nodes = {}
    edges = []
    with open(infile) as f:
        while True:
            l = f.readline()
            if not l:
                break
            elif 'label' in l and not '->' in l:
                splited = l.split()
                num = int(splited[0])
                labels = [x[x.find('=')+1:] for x in splited[1:] if 'label' in x]
                label = labels[0].strip('"\n')
                nodes[num] = label
            elif '->' in l:
                splited = l.split()
                ind = splited.index('->')
                edges.append((int(splited[ind-1]),int(splited[ind+1])))

    return nodes, edges

def main(args):
    infile  = args.infile
    #graph = nx.drawing.nx_pydot.read_dot(infile)
    nodes, edges = DOT2list(infile)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.keys())
    graph.add_edges_from(edges)
    #graph = nx.relabel_nodes(graph, nodes)

    
    nx.draw_networkx(graph)
    plt.show()

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
                        help='file name for genome input', default='log/graph')
    
    #parser.add_argument('-o', '--outPref', type=str,\
                        #help='file name prefix for result input', default='log/graph')

    #parser.add_argument('-d', '--default', type=str,\
                        #help='default hyperparameter file', default='p/default_wan.json')

    #parser.add_argument('-p', '--hyperparam', type=str,\
                        #help='hyperparameter file', default=None)
  
    #parser.add_argument('-v', '--view', type=str2bool,\
                        #help='Visualize trial?', default=False)

    args = parser.parse_args()
    main(args)
