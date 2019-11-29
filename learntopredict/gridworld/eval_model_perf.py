from model_grid import make_model as make_model_near
from model_grid_conv import make_model as make_model_conv
from model_grid_fc import make_model as make_model_fc

#from train_grid import simulate as simulate
from model_grid import simulate_with_acts as simulate

import numpy as np
from collections import namedtuple

Game = namedtuple('Game', ['env_name', 'time_factor', 'input_size', 'output_size', 'layers', 'activation', 'noise_bias', 'output_noise', 'experimental_mode'])

games = {}


apple_world_simple = Game(env_name='AppleWorldSimple',
                input_size=50,
                        output_size=5,
                                time_factor=0,
                                        layers=[100,10],
                                                activation='tanh',
                                                        noise_bias=0.0,
                                                                output_noise=[False,False,False],
                                                                        experimental_mode=True,
                                                                        )
import os

run_type = 'fc'

if run_type == 'fc':
    make_model = make_model_fc
    files = os.listdir('paper_data/fc/logs')
    pre_fix = 'paper_data/fc/logs' 
    files = os.listdir('log')
    pre_fix = 'log'
elif run_type =='conv':
    make_model = make_model_conv
    files = os.listdir('paper_data/conv_window/logs')
    pre_fix = 'paper_data/conv_window/logs'
elif run_type == 'near':
    make_model = make_model_near
    files = os.listdir('paper_data/near_window/logs')
    pre_fix = 'paper_data/near_window/logs'


files = list(filter(lambda x: 'best' in x and 'py' not in x, files))

#files = ['log/' + f for f in files]
#files = ['paper_data/near_window/logs/' + f for f in files]
#files = ['paper_data/conv_window/logs/' + f for f in files]
files = [pre_fix +'/'+ f for f in files]



games['apple_world_simple'] = apple_world_simple

game = apple_world_simple




def worker(model, seed, train_mode_int=1, max_len=-1):

  train_mode = (train_mode_int == 1)
  
  reward_list, t_list, actions_list = simulate(model,
    train_mode=train_mode, render_mode=False, num_episode=100, seed=seed, max_len=max_len)
  reward = np.mean(sorted(reward_list))
  t = np.mean(t_list)
  return reward, t, np.std(reward_list), reward_list, actions_list


#model = make_model_near(game)
#model = make_model_near(game)
model = make_model(game)

model.make_env(render_mode = True)
#model.load_model('log/p.1.0.apple_world_simple.CMAES.8.4.1556129012.7182245.best.json')
#model.load_model('log/p.0.0.apple_world_simple.CMAES.8.4.1556126892.9814494.best.json')

#print(worker(model, 0))


import os



diff = {}
#for f in files:
import copy

def process_file(f):
    diff = {}
    p = f.split('p.')[1].split('.apple')[0]
    p = float(p)
    if p not in diff:
        diff[p] = []
    model = make_model(game)
    model.load_model(f)
    model.make_env(render_mode=True)
    model.env.reset()
    #for direction in [0,1,2,3,4]:
    dir_data = {0: [],
                1: [],
                2: [],
                3: [],
                4: []}

    rew_data = worker(model,np.random.randint(1000000000))
    

    diff[p].append(rew_data[0])    
    return diff

import concurrent.futures

diff_dicts = []
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    out_dict = executor.map(process_file, files)
    diff_dicts.append(out_dict)

diff = {}
for d in list(out_dict):
    for k in d.keys():
        if k not in diff:
            diff[k]=[]
        diff[k] = diff[k] + d[k]

#print(diff)

for p in sorted(diff.keys()):
    print(p,"\t",np.mean(diff[p]),"\t",np.std(diff[p]))

 #print(p,"\t",np.mean([d[0] for d in diff[p]]),"\t",np.mean([d[1] for d in diff[p]]),"\t", np.mean()
