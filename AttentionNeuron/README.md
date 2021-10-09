# The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning

<img width="100%" src="https://media.giphy.com/media/OZHNm0MqNLCE4MtN9j/giphy.gif"></img>

This is the code for our AttentionNeuron [paper](https://arxiv.org/abs/2109.02869).
For more information, please refer to our [web site](https://attentionneuron.github.io/).

## Requirements

All required packages are listed in `requirements.txt`
```shell
pip install -r requirements.txt
```

## Evaluation

We provide pre-trained permutation invariant (PI) policies in the `pretrained` folder.  
```shell
# Test the policy in CartPoleSwingUp.
python eval_agent.py --log-dir=pretrained/cartpole_pi --n-episodes=100

# Test the policy in AntBulletEnv.
python eval_agent.py --log-dir=pretrained/ant_pi --n-episodes=100

# Test the policy in CarRacing (PI and non-PI policy).
python eval_agent.py --log-dir=pretrained/carracing_pi --n-episodes=100
python eval_agent.py --log-dir=pretrained/carracing --n-episodes=100

# Test the policy in PuzzlePong (behavior cloning, trained with 70% occlusion ratio).
python eval_agent.py --log-dir=pretrained/puzzle_pong --n-episodes=100
```

Each task has some configurations in the corresponding `config.gin` file that can be played with:

**Common to all tasks**

- `shuffle_on_reset` or `permute_obs` defines whether we shuffle the observations during rollouts.
- `render` defines whether to turn on visualization.
- `v` defines whether to be verbose.

**CartPoleSwingUp**

- `num_noise_channels` defines how many noisy channels to input to the agent.

**CarRacing**

- `bkg` defines the background to use. You can put a `.jpg` image in the `tasks/bkg` folder and use the image name.
E.g. `bkg="mt_fuji"` will load `tasks/bkg/mt_fuji.jpg` as the background. `bkg=None` refers to the original background.
- `patch_size` defines the size of the patch to be shuffled.

**PuzzlePong**

- `occlusion_ratio` defines the percentage of patches to discard during tests.
- `patch_size` defines the size of the patch to be shuffled.

## Training

For the paper, we trained our models on [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine) (GKE).  
To reduce your workload of setting up a GKE cluster, we provide code for local training.  
It can take longer to train the agents on a local machine, however it is possible to tune `--population-size`, `--reps`, etc to speed up.

```shell
# Train CartPoleSwingUp agent.
python train_agent.py --config=configs/cartpole_pi.gin \
--log-dir=log/cartpole_pi --reps=16 --max-iter=20000

# Train AntBullet agent.
python train_agent.py \
--config=configs/ant_pi.gin --log-dir=log/ant_pi --reps=16 --max-iter=20000

# Train CarRacing agent.
python train_agent.py --config=configs/carracing_pi.gin \
--log-dir=log/carracing_pi --reps=16 --max-iter=20000
```

## Citation

For attribution in academic contexts, please cite this work as

```
@incollection{attentionneuron2021,
  author    = {Yujin Tang and David Ha},
  title     = {The Sensory Neuron as a Transformer: Permutation-Invariant Neural Networks for Reinforcement Learning},
  booktitle = {Advances in Neural Information Processing Systems 34},
  year      = {2021},
  publisher = {Curran Associates, Inc.},
  url       = {https://attentionneuron.github.io},
  note      = "\url{https://attentionneuron.github.io}",
}
```

## Disclaimer

This is not an official Google product.