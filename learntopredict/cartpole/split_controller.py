import json

with open("log/learn_cartpole.pepg.16.384.best.json") as f:    
  data = json.load(f)
  controller = data[0][-71:]

with open("log/split_controller.json", 'wt') as out:
  res = json.dump([controller, 0], out, sort_keys=True, indent=0, separators=(',', ':'))
