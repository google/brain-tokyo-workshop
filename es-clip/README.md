## Modern Evolution Strategies for Creativity: Fitting Concrete Images and Abstract Concepts

<p align="left">
  <img width="50%" src="https://media.giphy.com/media/FsxTmSk3MIkDv7X6Eg/giphy.gif"></img>
</p>

This is the code for our work "Modern Evolution Strategies for Creativity: Fitting Concrete Images and Abstract Concepts".
For more information, please refer to <https://es-clip.github.io/>. 

**Update**: 

- 🌟NEW🌟 JAX-based (re)implementation on Colab
- Please scroll down for a section discribing community works.

## 🌟NEW🌟 JAX-based (Re)implementation on Colab

We have re-implemented ES-CLIP in JAX, which is more suitable for running on Colab as notebooks:

- https://github.com/google/evojax/blob/main/examples/notebooks/AbstractPainting01.ipynb
- https://github.com/google/evojax/blob/main/examples/notebooks/AbstractPainting02.ipynb

For the original implementation, see below.

## Requirements

1. Linux Operation System.
2. (Highly recommended) Multi-core CPU and high-end NVIDIA GPU(s). See section below for dicussion regarding the performance.
3. 64-bit Python 3.6 or higher installtion. We recommend [Anaconda3](https://www.anaconda.com/) or [venv](https://docs.python.org/3/library/venv.html) installtion for a easy setup.
4. Several Python Packages specified on `requirements.txt`.
5. [PyTorch](https://pytorch.org/) with CUDA support. We recommed referring to [its website](https://pytorch.org/) for instructions regarding installing PyTorch.
6. [PGPElib](https://github.com/nnaisense/pgpelib).
7. [CLIP](https://github.com/openai/CLIP) from OpenAI.

Below is an example of setting up on a Linux machine with CUDA installed using Anaconda3 for your reference. 
It installs the most recent version for all dependencies.
You may need to adjust accordingly for a CUDA-enabled PyTorch installation following instructions on <https://pytorch.org/>

```bash
conda create -n es-clip-code python=3.8 -y    # Requirement 3 - Python Installation.
pip install -r requirements.txt               # Requirement 4 - Python Packages.
pip install torch torchvision                 # Requirement 5 - PyTorch. 
conda install swig -y                         # Needed by PGPElib below
pip install 'git+https://github.com/nnaisense/pgpelib.git#egg=pgpelib'  # Requirement 6 - PEPGlib.
pip install git+https://github.com/openai/CLIP.git  # Requirement 7 - CLIP.
```

*Alternatively*, the snapshot of versions of dependencies is stored in `requirements.all_snapshot.txt` and can be used if you want to have exact versions of dependencies we used in development.

## Running Code

🌟  You can run the code with easy.  First, if you follow the conda setup, activate the conda environment by running 
```bash
conda activate es-clip-code
```

### Fitting a Bitmap

🌟 You can fit a bitmap by running 
```bash
python3 ./es_bitmap.py --target_fn assets/monalisa.png
``` 
By default, it fits the given image and output the evolution results to `./es_bitmap_out`. The computation would consume all available CPU cores, but does not require GPU(s).

The behavior of the fitting could be fine-tuned by specifying command line parameters, for example:
```bash
python3 ./es_bitmap.py \
    --target_fn assets/monalisa.png `# Filename of the target image` \
    --out es_bitmap_out             `# Output directory` \
    --height 200                    `# Canvas height` \
    --width  200                    `# Canvas width` \
    --n_triangle 50                 `# Number of triangles to use` \
    --n_population 256              `# Population size for ES algorithm` \
    --n_iterations 10000            `# Number of iterations for ES algorithm` \
    ;
```

For more options, consult the output of `python3 ./es_bitmap.py --help`.

### Fitting a Text Prompt as a Concept


🌟 You can fit a text prmopt by running 
```bash
python3 ./es_clip.py --prompt "Walt Disney World" --gpus 0
``` 
By default, it fits the given prompt and output the evolution results to `./es_clipt_out`, and uses the first GPU.

Like the previous script, The behavior of the fitting could be fine-tuned by specifying command line parameters, for example:

```bash
python3 ./es_clip.py \
    --prompt "Walt Disney World"    `# The prompt` \
    --out es_clip_out               `# Output directory` \
    --height 200                    `# Canvas height` \
    --width  200                    `# Canvas width` \
    --n_triangle 50                 `# Number of triangles to use` \
    --n_population 256              `# Population size for ES algorithm` \
    --n_iterations 2000             `# Number of iterations for ES algorithm` \
    --gpus 0 1 2 3 4 5 6 7          `# Uses 8 GPUs` \
    --thread_per_clip 2             `# Uses 2 threads per CLIP model`
    ;
```
Note that this script uses 8 GPUs in the example above.

For more options, consult the output of `python3 ./es_clip.py --help`.


## Performance

ES algorithm needs to evaluate the fitness for a pool of candidates. 
For an efficient processing, we implement a multi-processing computation for both fitting cases.
We also make our code benifit from using multiple GPUs for fitting a text by placing multiple CLIP models on GPUs.
While we develop on [GCP](https://cloud.google.com/) machine with 64 cores and 8 Tesla V100 GPUs,
We found that our code could also run with fewer resources.
Specifically,

- `es_bitmap.py` could effectively use multiple CPU cores, and has a performance roughly proportional to the number of CPU cores.
- `es_clip.py` could run reasonably well with one GPU while more GPUs are better. It could also run in a CPU-only setting, but that would be too slow for a reasonble expectation.


## Community Works

Here listed are some works by the community that are likely related to or based on this work. 
Please [let us know](https://github.com/es-clip/es-clip.github.io/issues) if you think some works are missing and should be added here!
**There are provided here for information purposes only and mentiong them does not consitute an endorsement, so proceed at your own discretion.**

* [@eyaler](https://twitter.com/eyaler) Provided a [biggan + clip + cma-es work](https://github.com/eyaler/clip_biggan) as a [Colab notebook](https://colab.sandbox.google.com/github/eyaler/clip_biggan/blob/main/WanderCLIP.ipynb).
* [@gestaltungai](https://twitter.com/gestaltungai) started a simple [Colab notebook](https://colab.research.google.com/drive/1DGNxs8E4cA_ZUwPQdusxDArCWj-JX5TG) to play with this work.

## Differentiable Baselines

Please see ``diff_baseline/*.ipynb`` for differentiable baselines. They are for reference purpose only.

## Disclaimer

This is not an official Google product.
