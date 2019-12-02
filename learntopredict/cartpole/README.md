# Learn M and C at the same time as one agent

Experiment trying to learn a world model of swing up cartpole

learn_cartpole.py - learn M and C together. C can see the actual observation 5 percent. of the time

dream_cartpole_swingup.py - taking the M from learn_cartpole and wrapping it into an env

cartpole_swingup.py - test environment to see if a C trained on dream_cartpole_swingup alone will work

# run pretrained models:

Run M + C experiment to visualize the world model in actual env (solid color means C is peeking)

```
python model.py learn_cartpole log/learn_cartpole.pepg.16.384.best.json
```

Confirm that the C learned in joint training works in actual env. `controller.json` is the last bit from  `learn_cartpole.pepg.16.384.best.json` that has the weights of the controller C.

```
python model.py cartpole_swingup log/controller.json
```

Controller trained inside world model M's reality:

```
python model.py dream_cartpole_swingup log/dream_cartpole_swingup.cma.1.32.best.json
```

Deploying policy learned in dream back to reality. Learns to swing up, but doesn't learn to stabilize once it is up (since the 5% glimpse is prob sufficient).

```
python model.py cartpole_swingup log/dream_cartpole_swingup.cma.1.32.best.json
```

# train models from scratch:

In `model.py` set `render_mode = False` and `final_mode = True` near the beginning of the file.

To train M+C together, on a beefy 96-core CPU box (i.e. on cloud):

```
python train.py learn_cartpole -n 96 -e 16 -t 4 -o pepg --sigma_init 0.1
```

But on a local desktop with 8-CPU cores only, use this command:

```
python train.py learn_cartpole -n 8 -e 1 -t 6 -o pepg --sigma_init 0.1
```

To train a controller from scratch inside of M's open-loop environment, run:

```
python train.py dream_cartpole_swingup -n 8 -e 1 -t 4 -o cma --sigma_init 0.5
```

Note that if you trained on a machine with less than 96 cores (i.e. 8 cores), you want to rename this line in `dream.py`:

`self.world_model.load_model("./log/learn_cartpole.pepg.16.384.best.json")`

to point to the correct `.json` file.
