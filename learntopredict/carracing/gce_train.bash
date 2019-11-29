xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py vae_racing_pure_world_linear -s 10

xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- python train.py vae_racing_world_linear -s 40
