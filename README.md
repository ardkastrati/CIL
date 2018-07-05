# Introduction

This is the project repository of the team **make** for the [Collaborative Filtering project][1] from the [Computational Intelligence Lab][2] at [ETH Zürich][3].

## Overview 
In this repository we provide implementations of the following collaborative filtering methods:
1. Stochastic Gradient Descent with Regularization (SGD) (as in CIL Exercise 4, Problem 3)
2. Bayesian Probabilistic Matrix Factorization (BPMF) (Salakhutdinov and Mnih, 2008)
3. Bayesian Probabilistic Mixture-Rank Matrix Factorization (BPMRMF) (our method)

For all other methods considered in our project report, we ran the implementations available in the [`Surprise`][4] library.

The model we developed is BPMFMR, in which we do a fully Bayesian treatment of the (hyper-)parameters of a Gaussian mixture model (GMM) to characterize user-item ratings as a mixture of LRMA models of different ranks. Please refer to our project report for more details.

## Installation
You can conveniently set up the environment using `conda` by running the following commands:
```console
$ git clone https://gitlab.ethz.ch/lming/make.git
$ cd make
$ conda env create
$ source activate make-env
```
(For details on how to install `conda`, please refer to the [official documentation][5].)

## Running
After creating and activating the `conda` environment, you can reproduce our results by simply running
```console
(make-env) $ python run.py
```

This will produce a file called `bpmrmf.csv` in the `submission` folder containing the test predictions of our model, which can then be submitted directly to the [Kaggle competition][2] to achieve the same score as we did with our selected submission.

## Configuration
You can conveniently change the type of model used and its configuration by altering the `config.py` file. In the following we will describe the details of this. The default parameters in the file will reproduce our final submission as well as the results reported in our paper (i.e., for each model).

### Structure of `config.py`
The configuration file contains five different `Python` dictionaries, one containing general parameters and one for the parameters of each model (i.e., NMF, SVD, SGD and BPMRMF).

### Choosing the Model
To choose the model, you can just set the dictionary entry
```python
general_params['model'] = "bpmrmf"
```
as desired. The options are: 'bpmrmf' (default), 'sgd', 'nmf', 'svd', 'svdpp'.

### Setting the General Parameters
The other general parameters (i.e., except for the model, which was described previously) are as follows:

Parameter | Default | Description
------------ | ------------- | -------------
`general_params['train_data_path']` | "data/data_train.csv" | Path to the training data
`general_params['test_data_path']` | "data/sampleSubmission.csv" | Path to the test data
`general_params['surprise_train_path']` | "data/data_train_surprise.csv" | Path to the training data in the format as required by the `Surprise` library
`general_params['n_users']` | 10000 | The number of users that are rating the items (The number of rows in the rating matrix)
`general_params['n_movies']` | 1000 | The number of items that are rated from the users (The number of columns in the rating matrix)
`general_params['train_pct']` | 1.0 | Percentage of the data to be used for training; the remaining data will be used for validation; thus, if set to 1.0, all the data will be used for training


### Setting the Parameters for BPMRMF
The parameters of BPMRMF (i.e., our method) are as follows:

Parameter &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Default | Description
------------ | ------------- | -------------
`n_features` | [8,9,10,50] | `list` of ranks of BPMRMF; if the list contains only one rank, it is equivalent to BPMF
`eval_iters` | 21 | number of MCMC iterations
`beta` | 2.0 | The precision of the gaussian distribution of the ratings
`beta0_user` | 2.0 | The hyperparameter of the Gaussian-Wishart distribution for the users. A multiplicative factor of the precision (the inverse of Gaussian) in the Gaussian that is sampled from the Wishart distribution.  
`beta0_item` | 2.0 | The hyperparameter of the Gaussian-Wishart distribution for the items. A multiplicative factor of the precision (the inverse of Gaussian) in the Gaussian that is sampled from the Wishart distribution.  
`nu0_user` | None | Degrees of freedom in the Wishart distribution for the users.
`nu0_item` | None | Degrees of freedom in the Wishart distribution for the items.
`mu0_user` | 0 | The mean of all the random means (for each user) of normally distributed U. If there is no prior information for this parameter, it is recommended to set it to 0, based on the symmetry argument.
`mu0_item` | 0 | he mean of all the random means (for each item) of normally distributed V. If there is no prior information for this parameter, it is recommended to set it to 0, based on the symmetry argument.
`max_rating` | 5. | The maximum rating number.
`min_rating` | 1. | The minimum rating number.
`tau` | 1. | The concetration hyperparameter of the Dirichlet distribution. If less then K (the number of different ranks) the mass will be highly concentrated in a few components, leaving the rest with almost no mass, meaning for each rating only a few ranks (or even only one rank) will be considered for the prediction.


## References
Salakhutdinov, Ruslan, and Andriy Mnih. "Bayesian probabilistic matrix factorization using Markov chain Monte Carlo." Proceedings of the 25th International Conference on Machine Learning. ACM, 2008.


## Authors
Ming-Da Liu Zhang, Ard Kastrati and Erik Alexander Daxberger


[1]: https://inclass.kaggle.com/c/cil-collab-filtering-2018
[2]: http://da.inf.ethz.ch/teaching/2018/CIL/
[3]: http://ethz.ch
[4]: http://surpriselib.com
[5]: https://conda.io/docs/user-guide/install/index.html
