<img align="right" width="100%" src="https://storage.googleapis.com/quickdraw-models/sketchRNN/attention/assets/card/CarRacingExt2SmallSize.gif">

&nbsp;

# CarRacing Variants

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingOriginal.gif">
<br/>

This repository contains various modifications of the `CarRacing-v0` OpenAI gym environment.  
Unlike other gym environments for RL, the environments are for generalization tests. E.g., one trains their RL agent in `CarRacing-v0` and then test the agent's performance in the modified environments here **WITHOUT** retraining.

## Installation

To install the modified CarRacing environments, you simply clone the repo and run the following command:
```
pip install -e ./CarRacingExtension
```

`test.py` serves as a visualization tool for you to see what the modified environments look like:
```
# `env-name` = {"Color", "Color3", "Bar", "Blob", "Noise", "Video"}, see details below.
python test.py --env-string {env-name}
```

## Environments and Leaderboard

All scores are averages over 100 consecutive tests.  

### CarRacingColor-v0

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingColor1.gif">

You can create this environment with `gym.make('CarRacingColor-v0')`.  
In this modification, we sample two scalar noises from the interval [-0.2, 0.2] and add them to the grass and lane color RGB vectors in `env.reset()`.

| Test only | Score in CarRacing-v0 | Score in CarRacingColor-v0 | Credit |
|-----------|-----------------------|----------------------------|---------
| Yes | <img src="https://render.githubusercontent.com/render/math?math=914 \pm 15"> | <img src="https://render.githubusercontent.com/render/math?math=866 \pm 112"> | [AttentionAgent](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=901 \pm 37"> | <img src="https://render.githubusercontent.com/render/math?math=655 \pm 353"> | [WorldModel](https://github.com/hardmaru/WorldModelsExperiments)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=865 \pm 159"> | <img src="https://render.githubusercontent.com/render/math?math=505 \pm 464"> | [PPO](https://github.com/xtma/pytorch_car_caring)
| Yes | <img src="https://render.githubusercontent.com/render/math?math=859 \pm 79"> | <img src="https://render.githubusercontent.com/render/math?math=442 \pm 362"> | [GA](https://github.com/sebastianrisi/ga-world-models)|

### CarRacingColor3-v0

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingColor3.gif">

You can create this environment with `gym.make('CarRacingColor3-v0')`.  
In this modification, we sample two 3D noise vectors from the interval [-0.2, 0.2] and add them to the grass and lane color RGB vectors in `env.reset()`.

| Test only | Score in CarRacing-v0 | Score in CarRacingColor3-v0 | Credit |
|-----------|-----------------------|----------------------------|---------
| Yes | <img src="https://render.githubusercontent.com/render/math?math=914 \pm 15"> | <img src="https://render.githubusercontent.com/render/math?math=673 \pm 372"> | [AttentionAgent](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=865 \pm 159"> | <img src="https://render.githubusercontent.com/render/math?math=579 \pm 444"> | [PPO](https://github.com/xtma/pytorch_car_caring)
| Yes | <img src="https://render.githubusercontent.com/render/math?math=901 \pm 37"> | <img src="https://render.githubusercontent.com/render/math?math=394 \pm 413"> | [WorldModel](https://github.com/hardmaru/WorldModelsExperiments)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=859 \pm 79"> | <img src="https://render.githubusercontent.com/render/math?math=160 \pm 304"> | [GA](https://github.com/sebastianrisi/ga-world-models)|

### CarRacingBar-v0

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingBar.gif">

You can create this environment with `gym.make('CarRacingBar-v0')`.  
In this modification, we add two vertical bars on the left and right side of such that the screen looks narrower.
However, no critical information is lost in this modification.

| Test only | Score in CarRacing-v0 | Score in CarRacingBar-v0 | Credit |
|-----------|-----------------------|----------------------------|---------
| Yes | <img src="https://render.githubusercontent.com/render/math?math=914 \pm 15"> | <img src="https://render.githubusercontent.com/render/math?math=900 \pm 35"> | [AttentionAgent](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=859 \pm 79"> | <img src="https://render.githubusercontent.com/render/math?math=675 \pm 254"> | [GA](https://github.com/sebastianrisi/ga-world-models)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=865 \pm 159"> | <img src="https://render.githubusercontent.com/render/math?math=615 \pm 217"> | [PPO](https://github.com/xtma/pytorch_car_caring)
| Yes | <img src="https://render.githubusercontent.com/render/math?math=901 \pm 37"> | <img src="https://render.githubusercontent.com/render/math?math=166 \pm 137"> | [WorldModel](https://github.com/hardmaru/WorldModelsExperiments)|


### CarRacingBlob-v0

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingBlob.gif">

You can create this environment with `gym.make('CarRacingBlob-v0')`.  
In this modification, we add a red blob at a fixed position in the car's frame.
Because the car runs counterclockwisely, we put the blob on the car's right side to reduce lane occlusion.

| Test only | Score in CarRacing-v0 | Score in CarRacingBlob-v0 | Credit |
|-----------|-----------------------|----------------------------|---------
| Yes | <img src="https://render.githubusercontent.com/render/math?math=914 \pm 15"> | <img src="https://render.githubusercontent.com/render/math?math=898 \pm 53"> | [AttentionAgent](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=865 \pm 159"> | <img src="https://render.githubusercontent.com/render/math?math=855 \pm 172"> | [PPO](https://github.com/xtma/pytorch_car_caring)
| Yes | <img src="https://render.githubusercontent.com/render/math?math=859 \pm 79"> | <img src="https://render.githubusercontent.com/render/math?math=833 \pm 135"> | [GA](https://github.com/sebastianrisi/ga-world-models)|
| Yes | <img src="https://render.githubusercontent.com/render/math?math=901 \pm 37"> | <img src="https://render.githubusercontent.com/render/math?math=446 \pm 299"> | [WorldModel](https://github.com/hardmaru/WorldModelsExperiments)|

### CarRacingNoise-v0

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingNoise.gif">

You can create this environment with `gym.make('CarRacingNoise-v0')`.  
In this modification, the grass background has been replaced with noise in RGB channels.
Notice that to learn a good RL agent in this environment is actually easy, for instance, one can sample muitple noises and add back to the observation to get a clearer observation.

| Test only | Score in CarRacing-v0 | Score in CarRacingNoise-v0 | Credit |
|-----------|-----------------------|----------------------------|---------
| Yes | <img src="https://render.githubusercontent.com/render/math?math=914 \pm 15"> | <img src="https://render.githubusercontent.com/render/math?math=-58 \pm 12"> | [AttentionAgent](https://github.com/google/brain-tokyo-workshop/tree/master/AttentionAgent)|

### CarRacingVideo-v0

<img align="right" width="200" src="https://github.com/lerrytang/public_resources/blob/main/CarRacingExtension/CarRacingKOF.gif">

You can create this environment with `gym.make('CarRacingVideo-v0')`.  
In this modification, the grass background has been replaced with frames from a user specified video.
The user is responsible for preparing the frames (E.g. with `ffmpeg`), naming them as `1.png`, `2.png`, etc.
When the frames are prepared, use one of the following commands to notify the environment.
We don't set leaderboard for this environment since different users can have different videos.
```
# Option 1. Put the directory in /tmp.
mv {frame-dir} /tmp/car_racing_video

# Option 2. Set environment variable.
export CARRACING_VIDEO_DIR="{frame-dir}"
```

## Citation
For attribution in academic contexts, please cite this work as

```
@inproceedings{attentionagent2020,
  author    = {Yujin Tang and Duong Nguyen and David Ha},
  title     = {Neuroevolution of Self-Interpretable Agents},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  url       = {https://attentionagent.github.io},
  note      = "\url{https://attentionagent.github.io}",
  year      = {2020}
}
```

## Disclaimer

This is not an official Google product.
