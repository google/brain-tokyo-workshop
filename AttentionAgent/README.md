# Neuroevolution of Self-Interpretable Agents

![attentionagent](https://storage.googleapis.com/quickdraw-models/sketchRNN/attention/assets/card/attentionagent.gif)  
Our agent receives visual input as a stream of 96x96px RGB images (left). Each image frame is passed through a self-attention bottleneck module, responsible for selecting K=10 patches (highlighted in white, middle). Features from these K patches (such as location) are then routed to a decision-making controller (right) that will produce the agent’s next action. The parameters of the self-attention module and the controller are trained together using neuroevolution.

This repository contains the code to reproduce the results presented in the orignal [paper](https://attentionagent.github.io/). 


## Dependencies

* CarRacing: run the command `pip3 install -r requirements.txt` and install all the required packages.
* DoomTakeCover: change `gym[box2d]==0.15.3` to `gym==0.9.4`, then run `pip3 install -r requirements.txt`.

Sometimes installing these dependencies locally can be challenging due to software incompatibility problems.  
We have prepared docker images to save your time, run the following commands to pull and connect to the image.  
```
# Download the image, {tag} can be "CarRacing" or "TakeCover".
docker image pull docker.io/braintok/self-attention-agent:{tag}

# Connect to the image, you can run the training/test commands in the container now.
docker run -it braintok/self-attention-agent:{tag} /bin/bash
```

## Evaluate pre-trained models

### Test models in the original environment
We have included our pre-trained models for both CarRacing and DoomTakeCover in this repository.  
You can run the following commands to evaluate the trained agent, change `pretrained/CarRacing` to `pretrained/TakeCover` to see results in DoomTakeCover.
```
# Evaluate for 100 episodes.
python3 test_solution.py --log-dir=pretrained/CarRacing/ --n-episodes=100

# Evaluate with GUI.
python3 test_solution.py --log-dir=pretrained/CarRacing/ --render

# Evaluate with GUI, also show attention patch visualization.
python3 test_solution.py --log-dir=pretrained/CarRacing/ --render --overplot

# Evaluate with GUI, save videos and screenshots.
python3 test_solution.py --log-dir=pretrained/CarRacing/ --render --overplot --save-screens
```

### Test models in the modified environments
Besides the original environment, we have tested our model in some modified environments in the paper.  
You can follow the instructions below to reproduce those results, the visual effect should be similar to those in the gifs.    

#### CarRacing 
![CarRacingExtensions](https://github.com/lerrytang/public_resources/blob/main/AttentionAgent/CarRacingVariants.gif)

The configuration file `pretrained/CarRacing/config.gin` serves as an effortless method to switch between environments.  
Change `utility.create_task.modification = "original"` to 
`utility.create_task.modification = "mod"` where `mod` is `color`, `bar` or `blob`.
Save the file, then run the following command:
```
python3 test_solution.py --log-dir=pretrained/CarRacing/ --render --overplot
```

#### DoomTakeCover
![TakeCoverExtensions](https://github.com/lerrytang/public_resources/blob/main/AttentionAgent/TakeCoverVariants.gif)

In `pretrained/TakeCover/config.gin`, change `utility.create_task.modification = "original"` to 
`utility.create_task.modification = "mod"` where `mod` is `wall`, `floor` or `text`.  
Save the file, then run the following command:
```
python3 test_solution.py --log-dir=pretrained/TakeCover/ --render --overplot
```

## Training

### Training locally
To train on a local machine or in a local container, run the following command:
```
# Train CarRacing locally with 4 worker processes.
bash train_local.sh -c configs/CarRacing.gin -n 4

# Train DoomTakeCover locally with 4 worker processes.
bash train_local.sh -c configs/TakeCover.gin -n 4
```

You can modify the corresponding `.gin` file to change parameters.  
For instance, changing `cma.CMA.population_size = 256` to `cma.CMA.population_size = 128` in `configs/CarRacing.gin` shrinks the population size to half.

### Training on a cluster

Running the code on a distributed computing platform such as Google Kubernetes Engine (GKE) can speed up the training a lot, see this [blog](https://cloud.google.com/blog/products/ai-machine-learning/how-to-run-evolution-strategies-on-google-kubernetes-engine) for more information about running evolution algorithms on GKE.  

Once you have a Kubernetes cluster ready (either locally or from a cloud service provider), you can package the code into an image and submit a job to the cluster.  
The following commands are only tested on GKE but should be general, steps 1 and 2 are *only* necessary for the first time or when you have made code modifications.

#### Step 1. Prepare an image.
You can use our prepared images `docker.io/braintok/self-attention-agent:{tag}`, in this case you can skip this step.  

If you decide to create your own image, you can run the following command in the repository's root directory.
```
docker build -t {image-name}:{tag} .
```

**Optional**  
Our code saves logs and model snapshots locally on the master node, this is not at all a problem if you are running on local machines. But on GKE, you need to download the data from the master node before deleting the cluster.
To get rid of this complexity, we support saving everything on Google Cloud Storage (GCS).  
If you happen to have a GCS bucket, you can create a credential json file from your GCP project, name it `gcs.json` and place it in the repository's root.  
The following instructions show how to add this json to an existing docker image.
```
# Connect to the docker image.
docker run -it {image-name}:{tag} /bin/bash

# Press ctrl+p and ctrl+q to detach from the container.

# Use the command to look up the container's id.
docker container ls

# Copy the json file to the container.
docker cp gcs.json {container-id}:/opt/app/

# Save the container as a new image.
docker container commit {container-id} {image-name}:{tag}
```

#### Step 2. Upload the image to docker hub or GCP.
Run the following commands to upload your image.
```
# Tag the image, you can look up the image name and version with `docker image ls`.
docker image tag {image-name}:{tag} docker.io/{your-docker-account-name}/self-attention-agent:{tag}

# Push the image to a remote registry.
docker image push docker.io/{your-docker-account-name}/self-attention-agent:{tag}
```

#### Step 3. Submit a job to Kubernetes.
We assume you have configured `kubectl` to work with your cluster. Otherwise, please consult this [doc](https://kubernetes.io/docs/tasks/tools/install-kubectl/) for setup instructions.

Once your `kubectl` works correctly with your cluster, you can run the following command to deploy a job on the cluster.
```
cd yaml/

# The following command submits a job to kubernetes, deploy_task.sh starts all workers first before the master.
# If this is the first time of deployment, kubernetes needs to download your image and this costs some time,
# we therefore set the wait-time to 180 seconds. Otherwise 30 seconds is usually enough.

# Run this command to deploy CarRacing training on a cluster. 
bash deploy_task.sh --config configs/CarRacing.gin --experiment-name training-car-racing --wait-time 180

# Modify deploy_task.sh such that IMAGE at line #6 points to TakeCover's image (E.g., self-attention-agent:TakeCover).
# Then run this command to deploy TakeCover training on a cluster.
bash deploy_task.sh --config configs/TakeCover.gin --experiment-name training-take-cover --wait-time 180
```

Congratulations! You have started your training task on your cluster.  
In the task deployment, we have also started an nginx process that allows you to check the training process via HTTP requests.
This is based the code in this [repository](https://github.com/lerrytang/es_on_gke), you can learn more about it there.

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
