<div align="center">

# Iterated Denoising Energy Matching for Sampling from Boltzmann Densities

[![Preprint](http://img.shields.io/badge/paper-arxiv.2402.06121-B31B1B.svg)](https://arxiv.org/abs/2402.06121)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

</div>

## Description

This is the official repository for the paper [Iterated Denoising Energy Matching for Sampling from Boltzmann Densities](https://arxiv.org/abs/2402.06121).

We propose iDEM, a scalable and efficient method to sample from unnormalized probability distributions. iDEM makes use of the DEM objective, inspired by the stochastic regression and simulation
free principles of score and flow matching objectives while allowing one to learn off-policy, in a loop while itself generating (optionally exploratory) new states which are subsequently
learned on. iDEM is also capable of incorporating symmetries, namely those represented by the product group of $SE(3) \\times \\mathbb{S}\_n$. We experiment on a 2D GMM task as well as a number of physics-inspired problems. These include:

- DW4 -- the 4-particle double well potential (8 dimensions total)
- LJ13 -- the 13-particle Lennard-Jones potential (39 dimensions total)
- LJ55 -- the 55-particle Lennard-Jones potential (165 dimensions total)

This code was taken from an internal repository and as such all commit history is lost here. Development credit for this repository goes primarily to [@atong01](https://github.com/atong01), [@jarridrb](https://github.com/jarridrb) and [@taraak](https://github.com/taraak) who built
out most of the code and experiments with help from [@sarthmit](https://github.com/sarthmit) and [@msendera](https://github.com/msendera). Finally, the code is based off the
[hydra lightning template](https://github.com/ashleve/lightning-hydra-template) by [@ashleve](https://github.com/ashleve) and makes use of the [FAB torch](https://github.com/lollcat/fab-torch) code for the GMM task and replay buffers.

## Installation

For installation, we recommend the use of Micromamba. Please refer [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) for an installation guide for Micromamba.
First, we install dependencies

```bash
# clone project
git clone git@github.com:jarridrb/DEM.git
cd DEM

# create micromamba environment
micromamba create -f environment.yaml
micromamba activate dem

# install requirements
pip install -r requirements.txt

```

Note that the hydra configs interpolate using some environment variables set in the file `.env`. We provide
an example `.env.example` file for convenience. Note that to use wandb we require that you set WANDB_ENTITY in your
`.env` file.

To run an experiment, e.g., GMM with iDEM, you can run on the command line

```bash
python dem/train.py experiment=gmm_idem
```

We include configs for all experiments matching the settings we used in our paper for both iDEM and pDEM except LJ55 for
which we only include a config for iDEM as pDEM had convergence issues on this dataset.

## Current Code

The current repository contains code for experiments for iDEM and pDEM as specified in our paper.

## Update January 2025
In this update we provide code and more detailed instructions on how to run the CFM models including log Z and ESS computation.
In doing this, we also found a few bugs in the public code implementation for LJ55 (note that this codebase is an adaptation of
a large number of notebooks used for the paper) which we have fixed in a set of code updates just merged to the repository.

### CFM for Computing NLL Pipeline
We will use the example of LJ55 in detailing the pipeline. First, run the training script as normal as follows

```bash
python dem/train.py experiment=lj55_idem
```

After training is complete, find the epochs with the best `val/2-Wasserstein` values in wandb. We will use the
best checkpoint to generate a training dataset for CFM in the following command. This command will also log the
2-Wasserstein and total variation distance for the dataset generated from the trained iDEM model compared to the
test set. To run this, you must provide the eval script with the checkpoint path you are using.

```bash
python dem/eval.py experiment=lj55_idem ckpt_path=<path_to_ckpt>
```

This will take some time to run and will generate a file named `samples_<n_samples_to_generate>.pt` in the hydra
runtime directory for the eval run. We can now use these samples to train a CFM model. We provide a config `lj55_idem_cfm`
which has the settings to enable the CFM pipeline to run by default for the LJ55 task, though doing so for other tasks
is also simple. The main config changes required are to set `model.debug_use_train_data=true, model.nll_with_cfm=true`
and `model.logz_with_cfm=true`. To point the CFM training run to the dataset generated from iDEM samples we can set the
`energy.data_path_train` attribute to the path of the generated samples. CFM training in this example can then be done
with

```bash
python dem/train.py experiment=lj55_idem_cfm energy.data_path_train=<path_to_samples>
```

Finally, to eval test set NLL, take the checkpoint of the CFM run with the best `val/nll` and run the eval script
again

```bash
python dem/eval.py experiment=lj55_idem_cfm ckpt_path=<path_to_cfm_ckpt>
```

Finally, we note that you may need to try a couple different checkpoints from the original
`python dem/train.py experiment=lj55_idem` run to be used in generating samples and downstream CFM training/eval in
order to get the best combination of eval metrics.

### ESS Computation Considerations
In preparing this update we noticed our original evaluation of ESS was evaluated on a batch size of 16 on all tasks. We recommend users of our
repository instead evaluate ESS on a larger batch size, (default to 1000) in the updated code. To reproduce the results in the paper you can
either set this to 16 or look at the wandb during validation when training the CFM model which evaluates on batch size 16.

### LJ55 negative time
In our original manuscript for LJ55 we used 10 steps of "negative time" (described in Section 4 of our manuscript)
during inference where we continued SDE inference for 10 extra steps using the true score at time 0. The repository
code had the flag to do this turned on in the configs but the code ignored this flag. This has been corrected in the update.

## Citations

If this codebase is useful towards other research efforts please consider citing us.

```
@misc{akhoundsadegh2024iterated,
      title={Iterated Denoising Energy Matching for Sampling from Boltzmann Densities},
      author={Tara Akhound-Sadegh and Jarrid Rector-Brooks and Avishek Joey Bose and Sarthak Mittal and Pablo Lemos and Cheng-Hao Liu and Marcin Sendera and Siamak Ravanbakhsh and Gauthier Gidel and Yoshua Bengio and Nikolay Malkin and Alexander Tong},
      year={2024},
      eprint={2402.06121},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contribute

We welcome issues and pull requests (especially bug fixes) and contributions.
We will try our best to improve readability and answer questions!

## Licences

This repo is licensed under the [MIT License](https://opensource.org/license/mit/).

### Warning: the current code uses PyTorch 2.0.0+

The code makes heavy use of the func torch library which is included in torch 2.0.0 as well as torch vmap.
