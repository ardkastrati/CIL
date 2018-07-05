## Introduction

This is the project repository of the team **make** for the [Collaborative Filtering project][1] from the [Computational Intelligence Lab][2] at [ETH Zürich][3].

## Overview 
We tried several models:
1. Stochastic Gradient Descent with Regularization (SGD).
2. Non-Negative Matrix Factorization (NMF).
3. Bayesian Probabilistic Matrix Factorization (BPMF).
4. Bayesian Probabilistic Matrix Factorization with Mixture Rank (BPMFMR).

Our best model is BPMFMR, in which we perform a fully Bayesian treatment of the (hyper-)parameters of a Gaussian mixture model (GMM) to characterize user-item ratings as a mixture of LRMA models of different ranks and propose a . 

## Installation
You can conveniently set up the environment using `conda` by running the following commands:
```console
$ git clone https://gitlab.ethz.ch/lming/make.git
$ cd make
$ conda env create
$ source activate make-env
```

## Running
After creating and activating the `conda` environment, you can reproduce our results by simply running
```console
(make-env) $ python run.py
```

This will produce a file called `bpmrmf.csv` in the `submission` folder containing the test predictions of our model, which can then be submitted directly to the [Kaggle competition][2] to achieve the same score as we did with our selected submission.

## Configuration
# Introduction
You can conveniently change the type of model used and its configuration by altering the `config.py` file. In the following we will describe the details of this.

# Structure of `config.py`
The configuration file contains five different `Python` dictionaries, one containing general parameters and one for the parameters of each model (i.e., NMF, SVD, SGD and BPMRMF).


## Authors
Ming-Da Liu Zhang, Ard Kastrati and Erik Alexander Daxberger


[1]: https://inclass.kaggle.com/c/cil-collab-filtering-2018
[2]: http://da.inf.ethz.ch/teaching/2018/CIL/
[3]: http://ethz.ch
