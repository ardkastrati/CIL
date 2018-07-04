# CIL

Project of team make for the Collaborative Filtering project from the Computational Intelligence Lab at ETH Zürich.

# Overview 
We tried several models:
1. Stochastic Gradient Descent with Regularization (SGD).
2. Non-Negative Matrix Factorization (NMF).
3. Bayesian Probabilistic Matrix Factorization (BPMF).
4. Bayesian Probabilistic Matrix Factorization with bias (BPMFB).
5. Bayesian Probabilistic Matrix Factorization with bias and implicit data (BPMFBI).
6. Bayesian Probabilistic Matrix Factorization with Mixture Rank (BPMFMR).

Our best model is BPMFMR, in which we employ a  Gaussian  mixture  model (GMM)  to characterize  user-item ratings as a mixture of LRMA models of  different ranks and propose a fully Bayesian treatment of the model (hyper-)parameters. 




# ToDo

- Ard 
	- Papers koren08 & he17
	- tune hyperparameters for NN and Koren
	- tune NN architecture
	- Simon Funk method
- Kou: Paper borgs17
- Ming: Paper zheng16
- Erik: 
	- Paper li17
	- try out some of the ideas below


# Timeline

| Week          | Tasks         |
| ------------- |:-------------:|
| Fr 22.06 - Fr 29.06 | do coding |
|     **30.06**     | **KAGGLE DEADLINE** |
| Sa 30.06 - Th 05.07 | write report  |
|     **06.07**     | **DEADLINE REPORT** |


# Ideas

- combine mixed-rank model with neural network model
- learn embeddings, cluster users+movies, run SVD with different ranks on submatrices
- run SVD multiple times (just to be sure)
- use an non-linear autoencoder (non-linear SVD)
- online learning (adpative, non-uniform sampling of training data)
- estimate a probability distribution over the ratings (as opposed to a point estimate) and exploit uncertainty information
- exploit other datasets (Netflix, MovieLens)
- learn distribution/embedding of (groups/clusters of) users/movies in our dataset and use it to extract useful information from public datasets
- train a generative model (sample in latent space) to augment training data
