# CIL

Project of team make for the Collaborative Filtering project from the Computational Intelligence Lab at ETH Zürich.

# Overview 
We tried several models:
1. Stochastic Gradient Descent with Regularization (SGD).
2. Non-Negative Matrix Factorization (NMF).
3. Bayesian Probabilistic Matrix Factorization (BPMF).
4. Bayesian Probabilistic Matrix Factorization with Mixture Rank (BPMFMR).

Our best model is BPMFMR, in which we employ a  Gaussian  mixture  model (GMM)  to characterize  user-item ratings as a mixture of LRMA models of  different ranks and propose a fully Bayesian treatment of the model (hyper-)parameters. 

# Running
You can reproduce our results by running:
python run.py

There is also a config.py file in which you can choose one of the implemented models, with different parameters.
The output submission will be a .csv file in the submission directory. For our model, it is called bpmrmf.csv.

# Authors
Ming-Da Liu Zhang, Ard Kastrati and Erik Alexander Daxberger


