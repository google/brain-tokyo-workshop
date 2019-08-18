# Weight Agnostic Neural Networks

Weight Agnostic Networks: network topologies evolved to work with a variety of shared weights. Adapted from the [prettyNEAT](../prettyNEAT) package. If you are just interested in seeing more details of how WANNs were implemented the [WANN](../WANN) repo is a cleaner self-contained version. This fork contains and adapts code used in the original NEAT (species, crossover, etc.) -- so if you are looking to extend or play around with WANNs you are in the right place.


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



## Training Weight Agnostic Neural Networks

To get started and see that everything is set up you can test the swing up domain:

```
python wann_train.py
```

which is the same as the default hyperparameter:

```
python wann_train.py -p p/laptop_swing.json -n 8
```

Where `8` is the number of workers you have on your computer, and `p/laptop_swing.json` contains hyperparameters.

Evaluation of the population is embarrassingly parallel so every extra core you have will really speed things up. Here is an example training curve on cart-pole with a population of 64 on an 8 core laptop:

![alt text](log/wann_run.png)

Where `Fitness` is the mean reward earned over all trials and weight values. `Median Fitness` is the fitness of the median performing member of the population, `Max Fitness` is the fitness of the best performing member of the population, `Top Fitness` is the best performing member ever found. `Peak Fitness` is the mean reward earned by the best performing member with its best performing weight value. To reproduce this graph see this [jupyter notebook](../WANN/log/viewRunStats.ipynb).

The full list of hyperparameters and their meaning is explained in [hypkey.txt](p/hypkey.txt)

## Testing and Tuning Weight Agnostic Neural Networks

To view or test a WANN:

```
python wann_test.py -p p/swingup.json -i champions/swing.out --nReps 3 --view True
```

WANNs are saved as 2D numpy arrays and can be retested, train, and viewed using the [WANNTool](../WANNTool) provided.

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
