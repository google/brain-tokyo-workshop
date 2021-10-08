<p align="center">
  <img width="100%" src="https://media.giphy.com/media/Q7IjOrLxlydlyb2kZL/giphy.gif">
</p>

# Weight Agnostic Neural Networks

Code to reproduce and extend the experiments in ['Weight Agnostic Neural Networks'](https://weightagnostic.github.io/) by Adam Gaier and David Ha. 

This repository is split into 4 parts:

* [WANN](WANN): Self-contained code for replicating the experiments in the paper. If you just want to look at the details of the implementation this is the code for you.

* [prettyNEAT](prettyNEAT): A general implementation of the NEAT algorithm -- used as an inspiration and departure point for WANNs. Performs simultaneous weight and topology optimization. If you want to do your own unrelated neuroevolution experiments with numpy and OpenAI Gym this is the code for you.

* [prettyNeatWann](prettyNeatWann): WANNs implemented as a fork of prettyNEAT -- inherits methods and structures from prettyNEAT. If you want to heavily modify or do extensive experiments with WANNs this is the code for you.

* [WANNTool](WANNTool): If you want to fine tune the weights of an existing WANN and test their performance over many trials, this is the code for you.

---

### Using the VAE Racer environment 

The pretrained VAE used in the VAERacer experiments is about 20MB, so rather than include it in every folder we put a single copy in the base directory. To use it copy the contents of `vae` into the root directory, e.g. 

`cp -r vae WANN/`

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
