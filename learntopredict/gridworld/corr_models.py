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

run_type = 'near'

if run_type == 'fc':
    make_model = make_model_fc
    files = os.listdir('log/')
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
  reward = np.mean(reward_list)
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
    action_data = np.asarray(rew_data[-1])
    unique, counts = np.unique(action_data, return_counts=True)
    dir_dict = dict(zip(unique,1.* counts/sum(counts)))
    #print(dir_dict)

    for k in range(100):
        
        for direction in [0,1,2,3,4]:
            model.make_env(render_mode=True)
            model.env.reset()
            obs= model.env.observe()
            new_obs = model.env.step(direction)
            new_obs = new_obs[0]
            if run_type == 'conv':
                sample = model.world_model.window_deterministic_predict_next_state(obs, direction)
            if run_type == 'near':
                sample = model.world_model.near_sight_deterministic_predict_next_state(obs, direction)
            if run_type == 'fc':
                sample = model.world_model.deterministic_predict_next_state(obs, direction)

            diff_val = ((new_obs.reshape(-1) - sample.reshape(-1))**2.).mean(axis=0)
            #if diff_val < 0.5:
            dir_data[direction].append(diff_val)
            #diff[p].append(diff_val)
    mins = []
    mins_probs = []
    for i in [0,1,2,3,4]:
        mins.append(np.mean(dir_data[i]))
        if i in dir_dict:
            mins_probs.append((np.mean(dir_data[i]), dir_dict[i]))
        else:
            mins_probs.append((np.mean(dir_data[i]), 0.0))
    cur_min = min(mins)

    index_of_min = 0
    for i,k in enumerate(mins):
        if k == cur_min:
            index_of_min = i
    index_of_max = 0
    cur_max = 0
    for i in range(5):
        if i in dir_dict:
            if  cur_max < dir_dict[i]:
                cur_max = dir_dict[i]
                index_of_max = i

    weighted_mins = 0
    for i in range(5):
        if i in dir_dict:
            weighted_mins+=dir_dict[i]*mins[i]
    #mins = min(mins)
    diff[p].append((cur_min,copy.deepcopy(rew_data[0]), mins_probs[index_of_max], weighted_mins))

    #mins = filter(lambda x: x!= cur_min, mins)

    #diff[p].append(min(mins))

    return diff

import concurrent.futures

diff_dicts = []
with concurrent.futures.ProcessPoolExecutor(max_workers=60) as executor:
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
    print(p,"\t",np.mean([d[3] for d in diff[p]]),"\t",np.std([d[3] for d in diff[p]]))

for p in sorted(diff.keys()):
    mean_0 = np.mean([d[2][0] for d in diff[p]])
    mean_1 = np.mean([d[2][1] for d in diff[p]])
    std_0 = np.std([d[2][0] for d in diff[p]])
    std_1 = np.std([d[2][1] for d in diff[p]])
    print(p,"\t", mean_0, "\t", mean_1, "\t", np.mean([(d[2][0]-mean_0)*(d[2][1]-mean_1)/(std_0*std_1) for d in diff[p]]))
    #print(p,"\t",np.mean([d[0] for d in diff[p]]),"\t",np.mean([d[1] for d in diff[p]]),"\t", np.mean()
