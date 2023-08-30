# Account Credibility Inference

This repository contains code to reproduce the results in the paper "Account credibility inference based on news sharing networks" by Bao Tran Truong, Oliver Melbourne Allen, Filippo Menczer.

## Project organization:

1. `data`: contains raw & derived datasets
2. `example`: minimal example 
3. `libs`: main module for the project
4. `paper`: experiment results and .ipynb noteboooks to reproduce figures
5. `workflow`: workflow files (Snakemake rules) and scripts

## Install 

- This code is written and tested with **Python=3.8.16**
- We use `conda`, a package manager to manage the development environment. Please make sure you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation) or [mamba](https://mamba.readthedocs.io/en/latest/installation.html#) installed on your machine

### Using Make (recommended)

To set up the environment and install the model: run `make` from the project directory (`credibility-inference`)

### Using Conda
1. Create the environment with required packages: run `conda env create -n credinference -f environment.yml` to 
2. Install the `credinference` package for the module imports to work correctly: 
    - activate virtualenv: `conda activate credinference`
    - run `pip install -e ./libs/` --- this use the
_"editable"_ option to install the package that does not require reinstallation if changes are made upstream. 
