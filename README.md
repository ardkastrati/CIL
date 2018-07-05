# CIL

SVD:
RMSE: 0.9965796535138055

NMF:
RMSE: 1.0007093783240044

SGD:
RMSE: 1.134257549370751

BPMF rank 8
36, train RMSE: 0.919386932791069 , val RMSE: 0.9812739446454559 

BPMF rank 9
iteration: 32, train RMSE: 0.9139997315475626 , val RMSE: 0.9808997873891879 

BPMF rank 10
iteration: 21, train RMSE: 0.9143092772462968 , val RMSE: 0.9818038027800366 

BPMF rank 50
iteration: 9, train RMSE: 0.8898841032499164 , val RMSE: 0.985430114045738 

BPMFMR
[8,9,10,50] 0.978407647369278

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


