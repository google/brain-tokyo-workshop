# WANN tool

WANN tool is an adaption of [estool](https://github.com/hardmaru/estool) to fine-tune the weights of WANNs for a particular tasks

## Dependencies

Core algorithm tested with:

- Python 3.5.3

- NumPy 1.15.2 (`pip install numpy`)

- mpi4py 3.0.1 (`pip install mpi4py`)

- OpenAI Gym 0.9.6 (`pip install gym` -- installation details [here](https://github.com/openai/gym))


Domains tested with:

- Cart-pole Swing-up (included, but requires OpenAI gym)

- Bipedal Walker: Box2d (see OpenAI gym installation)

- Quadruped (Ant) Walker: PyBullet 1.6.3 (`pip install pybullet`)

- MNIST: Mnist utilities 0.2.2 (`pip install mnist`)

- VAE Racer: 
    - Tensorflow 1.8 (`pip install tensorflow==1.8.0`)
    - Pretrained VAE (in [wannRelease](../) -- copy to root to use, e.g: `cp -r ../vae .`)


# how to run fine-tuned WANN models stored in zoo directory

To evaluate the pre-trained model on entire MNIST test set, once only:

`python model.py mnist256test -f zoo/mnist256.wann.json -e 1`

Should get 94.2% accuracy.

cartpole (100 times)

`python model.py cartpole_swingup -f zoo/cartpole_swingup.wann.json`

Should get 932 +/- 6

biped  (100 times)

`python model.py biped -f zoo/biped.wann.json`

Should get 332 +/- 1

vae_racing  (100 times - takes a very long time)

`python model.py vae_racing -f zoo/vae_racing.wann.json`

Should get 866 +/- 102

To visualize results, go to model.py, and make the two line change:

```
final_mode = False
render_mode = True
```

# how to fine-tune weights of an existing WANN

Below is an example of finetuning a WANN for cartpole swingup. The WANN for this environment is located in `champtions/swing.out` and this file is produced by WANN search from the main project repo.

`python train.py cartpole_swingup -e 4 -n 8 -t 4 -o cma --sigma_init 0.5`

The above command will launch 32 workers over 8 CPU cores (n=8, t=4), and evaluate each run (e=4) four times to calculate the reward. The optimization method used here is CMA-ES, with initial standard deviation for each individual weight set to 0.50. We fined that CMA-ES works better with a higher variance than the default optimizer based on population-based REINFORCE (`pepg`), which uses an initial standard deviation parameter of 0.10.

After 150 generations or so, which takes a tens of minutes on a macbook pro, the best agent should get an average score (over 128, 4x8x4 random rollouts) above 900 points, which will be fairly close to the best performance. You can hit `ctrl-c` to cancel the training job after 150 generations.

After the training, you should see the finetuned weight parameters stored in the `log` subdirectory. To run the WANN using the finetuned weights you just trained, use the following command:

`python model.py cartpole_swingup -f log/cartpole_swingup.cma.4.32.best.json`

This command will run the finetuned WANN model you just trained 100 times with different random seeds each time and you can confirm that the average score is > 900 over 100 runs.

If you set `render_mode = True` and `final_mode = False` in `model.py` you can visualize the finetuned WANN policy in action. The below command will run the model once if you want to visualize it after setting `render_mode` to `True` and `final_mode` to `False`:

`python model.py cartpole_swingup -f log/cartpole_swingup.cma.4.32.best.json -e 1`

The settings in this example is chosen to quickly finetune a WANN with minimal compute (i.e. on a personal laptop computer). For settings to reproduce results in the paper, please refer to the Appendix section.

---

### Citation
For attribution in academic contexts, please cite this work as

```
@article{wann2019,
  author = {Adam Gaier and David Ha},  
  title  = {Weight Agnostic Neural Networks},  
  eprint = {arXiv:1906.04358},  
  url    = {https://weightagnostic.github.io},  
  note   = "\url{https://weightagnostic.github.io}",  
  year   = {2019}  
}
```

## Disclaimer

This is not an official Google product.

