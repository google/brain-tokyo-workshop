import os

files = os.listdir('.')

best_files = []

for f in files:
    if 'best.' in f and 'py' not in f:
        best_files.append(f)


best_data = []

for f in best_files:
    with open(f) as ff:
        best_data.append(ff.readlines())

best_dict = {}

for b in best_data:
    #print(b)
    b = b[0]
    tab_split = b.split("\t")
    print(tab_split)
    peak = tab_split[0]

    best = tab_split[1]

    if float(peak) not in best_dict:
        best_dict[float(peak)] = [float(best)]
    else:
        best_dict[float(peak)].append(float(best))

print(best_dict)

import numpy as np

for k in sorted(best_dict.keys()):
    print(str(k) + '\t' + str(np.mean(best_dict[k])) + '\t' + str(np.std(best_dict[k])/np.sqrt(len(best_dict[k]))))
