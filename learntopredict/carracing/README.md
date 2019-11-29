# vae_racing

fork of estool where vae version of car_racing is added

## instructions

tested with tensorflow 1.8.0 cpu mode (for pre-trained vae)

use 64-core boxes on cloud, not 96-core boxes

## copy vae weights from WANN

From the WANN project, find `vae_16.json` and copy it into the vae subdirectory (same directory as `vae.py`)

## running pre-trained models:

The following command loads an agent trained to drive around the track from pixels (via VAE) with 30 percent peek probability:

```
python model.py learn_vae_racing log/learn_vae_racing.cma.4.64.best.json
```

This command loads an agent trained from scratch to drive, using only the 10 neuron activiations of the hidden layer of the world model learned by the previous agent (with a linear policy):

```
python model.py vae_racing_pure_world_linear log/vae_racing_pure_world_linear.cma.4.64.best.json
```

# train models from scratch:

On headless linux vms, need to put command `xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" --` in front of python call to launch headless X.

In `model.py` set `render_mode = False` and `final_mode = True` near the beginning of the file.

To train M+C together, on a beefy 64-core CPU box (i.e. on cloud):

```
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py learn_vae_racing
```

Train a linear controller from scratch using only the 10 hidden units of M (trained earlier): 

```
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py vae_racing_pure_world_linear
```
