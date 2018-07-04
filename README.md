# CIL

Implementations tried so far:

1. Stochastic Gradient Descent with Regularization (ex4-pyCharm)
2. Integrated model described in the "Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model" by Koren (Koren folder)
3. Neural Networks with user and movie embeddings (not finished yet, the architecture must be set and maybe use implicit data?) (keras folder)
4. Non-negative Matrix Factorization (nmf folder)
5. SVD (notebook folder)


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
