import numpy as np

def roulette(pArr):
  """Returns random index, with each choices chance weighted
  Args:
    pArr    - (np_array) - vector containing weighting of each choice
              [N X 1]

  Returns:
    choice  - (int)      - chosen index
  """
  spin = np.random.rand()*np.sum(pArr)
  slot = pArr[0]
  choice = len(pArr)
  for i in range(1,len(pArr)):
    if spin < slot:
      choice = i
      break
    else:
      slot += pArr[i]
  return choice

def listXor(b,c):
  """Returns elements in lists b and c they don't share
  """
  A = [a for a in b+c if (a not in b) or (a not in c)]
  return A

def rankArray(X):
  """Returns ranking of a list, with ties resolved by first-found first-order
  NOTE: Sorts descending to follow numpy conventions
  """ 
  tmp = np.argsort(X)
  rank = np.empty_like(tmp)
  rank[tmp] = np.arange(len(X))
  return rank

def tiedRank(X):  
  """Returns ranking of a list, with ties recieving and averaged rank
  # Modified from: github.com/cmoscardi/ox_ml_practical/blob/master/util.py
  """
  Z = [(x, i) for i, x in enumerate(X)]  
  Z.sort(reverse=True)  
  n = len(Z)  
  Rx = [0]*n   
  start = 0 # starting mark  
  for i in range(1, n):  
     if Z[i][0] != Z[i-1][0]:
       for j in range(start, i):  
         Rx[Z[j][1]] = float(start+1+i)/2.0;
       start = i
  for j in range(start, n):  
    Rx[Z[j][1]] = float(start+1+n)/2.0;

  return np.asarray(Rx)

def bestIntSplit(ratio, total):
  """Divides a total into integer shares that best reflects ratio
    Args:
      share      - [1 X N ] - Percentage in each pile
      total      - [int   ] - Integer total to split
    
    Returns:
      intSplit   - [1 x N ] - Number in each pile
  """
  # Handle poorly defined ratio
  if sum(ratio) is not 1:
    ratio = np.asarray(ratio)/sum(ratio)
  
  # Get share in real and integer values
  floatSplit = np.multiply(ratio,total)
  intSplit   = np.floor(floatSplit)
  remainder  = int(total - sum(intSplit))
  
  # Rank piles by most cheated by rounding
  deserving = np.argsort(-(floatSplit-intSplit),axis=0)
  
  # Distribute remained to most deserving
  intSplit[deserving[:remainder]] = intSplit[deserving[:remainder]] + 1    
  return intSplit

def quickINTersect(A,B):
  """ Faster set intersect: only valid for vectors of positive integers.
  (useful for matching indices)
    
    Example:
    A = np.array([0,1,2,3,5],dtype=np.int16)
    B = np.array([0,1,6,5],dtype=np.int16)
    C = np.array([0],dtype=np.int16)
    D = np.array([],dtype=np.int16)

    print(quickINTersect(A,B))
    print(quickINTersect(B,C))
    print(quickINTersect(B,D))
  """
  if (len(A) is 0) or (len(B) is 0):
    return [],[]
  P = np.zeros((1+max(max(A),max(B))),dtype=bool)
  P[A] = True
  IB = P[B]
  P[A] = False # Reset
  P[B] = True
  IA = P[A]

  return IA, IB