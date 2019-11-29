This branch contains files necessary to the apple world set of experiments.

USAGE:

To generate data equivalent to that used in this paper, run (preferably on a beefy machine):

nohup bash launch_simple.sh > launch_progress.out &

IMPORTANT: To change whether you're running a fully connected world model or a convolutional world model, change the import at the top of train_grid.py appropriately (see the comment in the file).


Contents:


apple_world_simple.py: A gym-like environment definition for the grid world environment. 

corr_models.py: A script used to generate the correlation statistics visualized in, e.g., Figures 4, 6, 9, and 10 in the paper.

eval_models_perf.py: A script used to generate evaluation performance of trained models.

log: An empty log directory.  Will be populated upon executing launch_simple.sh

model_params_count.py: A script to verify the parameter counts of the models.

setup.py: Setup for some dependencies.

config.py: Configuration definitions for the environments.

env.py: Definitions for the environmental make_env fn.

model_grid_fc.py: Code defining the fully connected world model, the policy, and other convenience functions.  Mostly duplicate code with model_grid_near, except for the parts changing the definition of the world model.    

model_grid_near.py: Code defining the convolutional world model.  Again, mostly duplicate with model_grid_fc.  

parse_best.py: Script for extracting useful data from the best runs during training.

train_grid.py: Contains the bulk of the training logic. IMPORTANT! If you want to train a fully connected model vs a convolutional model MAKE SURE TO CHANGE THE IMPORT AT THE TOP.

core.py: Thin, fully pythonic grid world library.

launch_simple.sh: Launch script for experiments.

README.md: This file.


