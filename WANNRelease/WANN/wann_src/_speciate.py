# -- Species container for population ------------------------------------ -- #
"""
In WANNs we aren't using species (for simplicity -- nothing forbids 
their use) but the code was adapted from the prettyNEAT package, whose 
operators act on a population in Species class containers. So here we 
dump the entire population into a Species.
"""


class Species():
  """Species class, only contains fields: all methods belong to WANN class.
  """
  def __init__(self,seed):
    """Initialize species around a seed
    Args:
      seed - (Ind) - individual which anchors seed in compatibility space

    Attributes:
      seed       - (Ind)   - individual who acts center of species
      members    - [Ind]   - individuals in species
      bestInd    - (Ind)   - highest fitness individual ever found in species
      bestFit    - (float) - highest fitness ever found in species
      lastImp    - (int)   - generations since a new best individual was found
      nOffspring - (int)   - new individuals to create this generation
    """
    self.seed = seed      # Seed is type Ind
    self.members = [seed] # All inds in species
    self.bestInd = seed
    self.bestFit = seed.fitness
    self.lastImp = 0
    self.nOffspring = []

def speciate(self):  
  """Divides population into species and assigns each a number of offspring

  NOTE: In WANNs we aren't using speciation -- so we just put all individuals
  in the same species, this is to just to fit things into the prettyNEAT code
  """
  self.species = [Species(self.pop[0])]
  self.species[0].nOffspring = self.p['popSize']
  for ind in self.pop:
    ind.species = 0
  self.species[0].members = self.pop
