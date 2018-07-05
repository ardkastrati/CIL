"""
Reference papers: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.
                 25th International Conference on Machine Learning (ICML-2008)
                Li, Dongsheng, et al. "Mixture-Rank Matrix Approximation for Collaborative Filtering.
                 "Advances in Neural Information Processing Systems. 2017.
Implementation of bpmf borrowed from: https://github.com/chyikwei/recommend, restructured and adapted for the BPMRMF.
"""
import logging
from six.moves import xrange
import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart, norm

from MRMF_BPMF.utils.base import ModelBase
from MRMF_BPMF.utils.exceptions import NotFittedError
from MRMF_BPMF.utils.datasets import build_user_item_matrix
from MRMF_BPMF.utils.validation import check_ratings
from MRMF_BPMF.utils.evaluation import RMSE
import time
logger = logging.getLogger(__name__)

class BPMRMF(ModelBase):
    """
    Bayesian Probabilistic Matrix Factorization with Mixture Ranking

    Initializes the BPMRMF.

    :param n_user: The number of users that are rating the items (The number of rows in the matrix)
    :param n_item: The number of items that are rated from the users (The number of columns in the matrix)
    :param n_feature: The ranks of BPMRF.
    :param beta: The precision of the gaussian distribution of the ratings (The inverse of :math:`\sigma^2`)
    :param beta0_user: The hyperparameter of the Gaussian-Wishart distribution for the users. A multiplicative factor of
                       the precision (the inverse of Gaussian) in the Gaussian that is sampled from the Wishart distribution.
    :param nu0_user:  Degrees of freedom in the Wishart distribution for the users.
    :param mu0_user: The mean of all the random means (for each user) of normally distributed :math:`U`. If there is no
                     prior information for this parameter, it is recommended to set it to 0, based on the symmetry argument.
    :param beta0_item: The hyperparameter of the Gaussian-Wishart distribution for the items. A multiplicative factor of
                       the precision (the inverse of Gaussian) in the Gaussian that is sampled from the Wishart distribution.
    :param nu0_item: Degrees of freedom in the Wishart distribution for the items.
    :param mu0_item: he mean of all the random means (for each item) of normally distributed :math:`V`. If there is no
                     prior information for this parameter, it is recommended to set it to 0, based on the symmetry argument.
    :param seed: The seed. If one needs to have the same output for each run, then seed must be set to a fixed number.
    :param tau: The concetration hyperparameter of the Dirichlet distribution. If less then :math:`K` (the number of different ranks)
                the mass will be highly concentrated in a few components, leaving the rest with almost no mass, meaning for each rating
                only a few ranks (or even only one rank) will be considered for the prediction.
    :param max_rating: The maximum rating number.
    :param min_rating: The minimum rating number.
    """
    def __init__(self, n_user, n_item, n_feature=[8], beta=2.0,
                 beta0_user=2.0, nu0_user=None, mu0_user=0.,
                 beta0_item=2.0, nu0_item=None, mu0_item=0.,
                 seed=None, tau=1.,
                 max_rating=None,min_rating=None):

        super(BPMRMF, self).__init__()
        #General data
        self.n_user = n_user # number of users
        self.n_item = n_item # number of movies
        self.n_feature = n_feature # rank of the matrix
        self.K = len(n_feature) # number of models
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.tau = tau #Dirichlet parameter

        # Hyper Parameter
        self.beta = beta # The sigma in the normal function

        # Inv-Wishart (User features = U over each k)
        self.W0_user =  [np.eye(n_feature[k], dtype='float64') for k in range(self.K)]
        self.beta0_user = [beta0_user for _ in range(self.K)]
        self.nu0_user = n_feature
        self.mu0_user = [np.repeat(mu0_user, n_feature[k]).reshape(n_feature[k], 1) for k in range(self.K)]

        # Inv-Wishart (item features = V over each k)
        self.W0_item = [np.eye(n_feature[k], dtype='float64') for k in range(self.K)]
        self.beta0_item = [beta0_item for _ in range(self.K)]
        self.nu0_item = n_feature
        self.mu0_item = [np.repeat(mu0_item, n_feature[k]).reshape(n_feature[k], 1) for k in range(self.K)]

        # Latent Variables
        self.mu_user = [np.zeros((n_feature[k], 1), dtype='float64') for k in range(self.K)]
        self.mu_item = [np.zeros((n_feature[k], 1), dtype='float64') for k in range(self.K)]

        self.alpha_user = [np.eye(n_feature[k], dtype='float64') for k in range(self.K)]
        self.alpha_item = [np.eye(n_feature[k], dtype='float64') for k in range(self.K)]

        # Probabilities and classes
        self.user_class_probability_ = np.zeros((self.K, self.n_user))
        self.item_class_probability_ = np.zeros((self.K, self.n_item))
        self.classes = None

        # initializes the features randomly.
        # (There is no special reason to use 0.3)
        self.user_features_ = [0.3 * self.rand_state.rand(n_user, n_feature[k]) for k in range(self.K)]
        self.item_features_ = [0.3 * self.rand_state.rand(n_item, n_feature[k]) for k in range(self.K)]

        # data states
        self.iter_ = 0
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None
        self.predictions = None
        self.train_preds = None
        self.validation_preds = None
        self.MCMC_iteration = 0

    def MCMC(self, ratings, validation, test):
        """
        Obtain predictions that minimize the RMSE using Bayes estimator with MSE. More information in the report (Eq. (5)).
        The ratings, validation, test parameters are needed for performance reasons. We predict only the ratings that we need,
        and not all the ratings in the matrix.

        :param ratings: Training data. It is a numpy array matrix of shape (n_users*n_items, 3).
        Each element has the form [user_id, item_id, rating]
        :param validation: Validation data, it has the same format as ratings
        :param test: Test data, it has the same format as the training/validation data. The third column is just a dummy rating.
        """

        iteration = self.MCMC_iteration
        train_preds = self.__predict_in_current_iteration(ratings[:, :2])
        self.train_preds *= (iteration / (iteration + 1))
        self.train_preds += train_preds / (iteration + 1)
        train_rmse = RMSE(self.train_preds, ratings[:, 2])

        validation_preds = self.__predict_in_current_iteration(validation)
        self.validation_preds *= (iteration / (iteration + 1))
        self.validation_preds += validation_preds / (iteration + 1)
        val_rmse = RMSE(self.validation_preds, validation[:, 2])

        predictions = self.__predict_in_current_iteration(test)
        self.predictions *= (iteration / (iteration + 1))
        self.predictions += predictions / (iteration + 1)

        print("iteration: {}, train RMSE: {} , val RMSE: {} ".format(self.iter_, train_rmse, val_rmse))

        self.MCMC_iteration += 1


    def fit(self, ratings, validation, test, n_iters=50):
        """
        Train the model by using MCMC approximation technique. Gibbs sampling is used to sample the parameters for each iteration.
        The ratings, validation, test parameters are needed for performance reasons. We predict only the ratings that we need,
        and not all the ratings in the matrix. More information can be found in the report (Eq. (4) and (5)).

        :param ratings: Training data. It is a numpy array matrix of shape (n_users*n_items, 3).
        Each element has the form [user_id, item_id, rating]
        :param validation: Validation data, it has the same format as ratings
        :param test: Test data, it has the same format as the training/validation data. The third column is just a dummy rating.
        :param n_iters: number of iterations to perform Gibbs sampling.
        """

        # Check correctness of the ratings matrix
        check_ratings(ratings, self.n_user, self.n_item, self.max_rating, self.min_rating)

        #Initialize ratings for MCMC
        self.predictions = np.zeros(len(test))
        self.train_preds = np.zeros(len(ratings))
        self.validation_preds = np.zeros(len(validation))

        # Initialize the classes randomly
        self.classes = self.rand_state.randint(0, self.K, size=len(ratings))
        self.build_rating_matrices(ratings, self.classes)

        self.mean_rating_ = np.mean(ratings[:, 2])

        # Perform gibbs sampling for n_iters
        for iteration in xrange(n_iters):

            # Gibbs Sampling
            #Sample alpha and beta probability
            self.user_class_probability_ = self.sample_user_class_probability()

            self.item_class_probability_ = self.sample_item_class_probability()
            # Sample the classes
            self.classes = self.sample_user_item_classes(ratings)
            # Sampling hyperparameters
            self.mu_item, self.alpha_item = self.sample_item_params()
            self.mu_user, self.alpha_user = self.sample_user_params()
            # Sampling features
            self.item_features_ = self.sample_item_features()
            self.user_features_ = self.sample_user_features()
            # Build rating matrices based on their respective class = R(k)
            self.build_rating_matrices(ratings, self.classes)

            #Do MCMC (only for predictions)
            self.MCMC(ratings, validation, test)

            self.iter_ += 1
            print("Timestamp: {}".format(time.time()))
        return self

    def __predict_in_current_iteration(self, data):
        """
        Predict the data based on the current sampled parameters in the MCMC.

        :param data: Data to predict. It is a numpy array matrix of shape (number of data, 3).
        Each element has the form [user_id, item_id, rating].
        :returns: predicted data.
        """
        if not self.mean_rating_:
            raise NotFittedError()

        preds = 0
        # Get the weights for each model used
        class_probabilities = self.user_class_probability_.take(data.take(0, axis=1), axis=1) * \
                              self.item_class_probability_.take(data.take(1, axis=1), axis=1)

        sum = class_probabilities.sum(axis=0)
        class_probabilities[:, sum == 0] = 0.5
        class_probabilities /= class_probabilities.sum(axis=0)

        # Get features for each model and weighted prediction
        for k in range(self.K):
            u_features = self.user_features_[k].take(data.take(0, axis=1), axis=0)
            i_features = self.item_features_[k].take(data.take(1, axis=1), axis=0)
            val = np.sum(u_features * i_features, axis=1)

            preds += val*class_probabilities[k, :]

        preds += self.mean_rating_
        # Clip the prediction
        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def predict(self):
        """
        Returns the predictions for the test_data given in the during the fit by using MCMC (more information in Eq. (5) in the report).
        :return: the predictions of the test data.
        """
        return self.predictions

    def sample_user_class_probability(self):
        """
        Sample user class probabilities (:math:`\alpha` in the report, Section III, C) Inference) from the Dirichlet distribution.
        :return (:math:`\alpha`)
        """
        occurences = np.array([self.ratings_csr_[k].getnnz(axis=1) for k in range(self.K)], dtype=float)
        occurences += self.tau/self.K
        occurences /= (occurences.sum(axis=0) + self.tau)
        user_class_probabilities = np.array([self.rand_state.dirichlet(occurences[:, user_id]) for user_id in range(self.n_user)]).T
        return user_class_probabilities


    def sample_item_class_probability(self):
        """
        Sample user class probabilities (:math:`\beta` in the report, Section III, C) Inference) from the Dirichlet distribution
        :return returns user class probabilities (:math:`\beta`)
        """
        occurences = np.array([self.ratings_csr_[k].getnnz(axis=0) for k in range(self.K)], dtype=float)
        occurences += self.tau / self.K
        occurences /= (occurences.sum(axis=0) + self.tau)
        item_class_probabilities = np.array([self.rand_state.dirichlet(occurences[:, item_id]) for item_id in range(self.n_item)]).T
        return item_class_probabilities

    def sample_user_item_classes(self, ratings):
        """
        Sample classes probabilities (:math:`\c_{i,j}`) in the report, Section III, C) Inference)
        :param ratings: Training data. It is a numpy array matrix of shape (number of ratings, 3).
        Each element has the form [user_id, item_id, rating]
        :return the classes for each rating
        """

        discrete_probability = np.zeros((self.K, len(ratings)))

        alpha_prob = self.user_class_probability_.take(ratings.take(0, axis=1), axis=1)
        beta_prob = self.item_class_probability_.take(ratings.take(1, axis=1), axis=1)

        for k in range(self.K):
            #Find means
            u_features = self.user_features_[k].take(ratings.take(0, axis=1), axis=0)
            i_features = self.item_features_[k].take(ratings.take(1, axis=1), axis=0)
            val = np.sum(u_features * i_features, axis=1)
            val = ratings[:, 2] - val
            #Find probabilities
            probabilities = norm(0, self.beta).pdf(val)
            final_prob = probabilities * alpha_prob[k, :] * beta_prob[k, :]

            discrete_probability[k] = final_prob
        #Normalize
        sum = discrete_probability.sum(axis=0)
        discrete_probability[:, sum == 0] = 1./self.K
        discrete_probability /= discrete_probability.sum(axis=0)
        #sample from discrete
        classes = np.array([self.rand_state.multinomial(n=1, pvals=discrete_probability[:, i]).argmax() for i in range(len(ratings))])
        return classes

    def sample_user_params(self):
        """
        Sample user hyperparameters (:math:`\{\Theta_{U^k}\}_{k=1}^{n_users}` in the report, Section III, C) Inference) from the Gaussian-Wishart.
        :return :math:`\{\Theta_{U^k} = \{\mu_{U^k}, \Lambda_{V^k}\}\}_{k=1}^{n_items}`
        """
        alpha_user = []
        mu_user = []
        # Sample for every model
        for k in range(self.K):
            mu0_star, beta0_star, nu0_star, W0_star = \
                self.bayesian_update(self.n_user, self.mu0_user[k], self.beta0_user[k], self.nu0_user[k], self.W0_user[k], self.user_features_[k], self.n_feature[k])

            alpha_user.append(self.sample_Wishart(W0_star, nu0_star))
            mu_user.append(self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_user[k]))))
        return mu_user, alpha_user


    def sample_item_params(self):
        """
        Sample item hyperparameters (:math:`\{\Theta_{V^k}\}_{k=1}^{n_items}` in the report, Section III, C) Inference) from the Gaussian-Wishart.
        :return :math:`\{\Theta_{V^k} = \{\mu_{U^k}, \Lambda_{V^k}\}\}_{k=1}^{n_items}`
        """
        alpha_item = []
        mu_item = []
        # Sample for every model
        for k in range(self.K):
            mu0_star, beta0_star, nu0_star, W0_star = \
                self.bayesian_update(self.n_item, self.mu0_item[k], self.beta0_item[k], self.nu0_item[k], self.W0_item[k], self.item_features_[k], self.n_feature[k])

            alpha_item.append(self.sample_Wishart(W0_star, nu0_star))
            mu_item.append(self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_item[k]))))
        return mu_item, alpha_item


    def sample_user_features(self):
        """
        Sample user latent features (:math:`\{U^k\}_{k=1}^{n_users}` in the report, Section III, C) Inference) from the Gaussian distribution.
        :return :math:`\{U^k\}_{k=1}^{n_users}`
        """
        user_features_ = []
        for k in range(self.K):
            curr_user_features_ = np.zeros((self.n_user, self.n_feature[k]), dtype='float64')
            # Sample parameters for every user
            for user_id in xrange(self.n_user):
                indices = self.ratings_csr_[k][user_id, :].indices
                features = self.item_features_[k][indices, :]

                rating = self.ratings_csr_[k][user_id, :].data - self.mean_rating_
                rating = np.reshape(rating, (rating.shape[0], 1))

                mu_star, alpha_star_inv = self.conjugate_prior(self.mu_user[k], self.alpha_user[k], features, rating)
                # Sample user features from a gaussian distribution
                curr_user_features_[user_id] = self.sample_Gaussian(mu_star, alpha_star_inv)

            user_features_.append(curr_user_features_)

        return user_features_

    def sample_item_features(self):
        """
        Sample user latent features (:math:`\{V^k\}_{k=1}^{n_items}` in the report, Section III, C) Inference) from the Gaussian distribution.
        :return :math:`\{V^k\}_{k=1}^{n_items}`
        """
        item_features_ = []
        for k in range(self.K):
            curr_item_features_ = np.zeros((self.n_item, self.n_feature[k]), dtype='float64')
            # Sample parameters for every movie
            for item_id in xrange(self.n_item):
                indices = self.ratings_csc_[k][:, item_id].indices
                features = self.user_features_[k][indices, :]

                rating = self.ratings_csc_[k][:, item_id].data - self.mean_rating_
                rating = np.reshape(rating, (rating.shape[0], 1))

                mu_star, alpha_star_inv = self.conjugate_prior(self.mu_item[k], self.alpha_item[k], features, rating)
                # Sample movie features from a gaussian distribution
                curr_item_features_[item_id] = self.sample_Gaussian(mu_star, alpha_star_inv)

            item_features_.append(curr_item_features_)

        return item_features_

    def bayesian_update(self, N, mu0, beta0, nu0, W0, evidence, evidence_n_feature):
        """
        Update the conditional parameters of the Gaussian-Wishart distribution. More information in the report Section III, C) Inference.
        Given a prior and new evidence this updates the parameters to get the posterior for the Gaussian Wishart.
        :param N: int. Number of samples that constituent the evidence
        :param mu0: the prior mean to be updated
        :param beta0: the prior multiplicative factor to be updated
        :param nu0: the prior number of degrees of freedom to be updated
        :param W0: the prior scale matrix to be updated
        :param evidence: the new evidence
        :param evidence_n_feature: number of features per evidence.
        :returns: the updated parameters mu0, beta0, nu0 and W0
        """
        X_bar = np.mean(evidence, 0).reshape((evidence_n_feature, 1))
        S_bar = np.cov(evidence.T)
        diff_X_bar = mu0 - X_bar
        W0_star = inv(inv(W0) +  N * S_bar + np.dot(diff_X_bar, diff_X_bar.T) * (N * beta0) / (beta0 + N))

        W0_star = (W0_star + W0_star.T) / 2.0

        nu0_star = nu0 + N
        beta0_star = beta0 + N
        mu0_star = (beta0 * mu0 + N*X_bar) / (beta0 + N)

        return mu0_star, beta0_star, nu0_star, W0_star

    # Calculate conjugate prior for the parameters
    def conjugate_prior(self, mu, alpha, features, rating):
        """
        Finds the hyperparameters of Gaussian-Wishart conjugate prior. More information in the report Section III,
        C) Inference (Latent features U and V).
        :param mu: the prior mean of the Gaussian distribution
        :param alpha: the precision of the gaussian distribution (The inverse of :math:`\sigma^2`).
        :param features: The features for the update
        :param rating: the ratings.
        :return: the updated mean and precision for the posterior Gaussian distribution.
        """
        alpha_star = alpha + self.beta * np.dot(features.T, features)
        alpha_star_inv = inv(alpha_star)
        temp = (self.beta * np.dot(features.T, rating).T + np.dot(alpha, mu))
        mu_star = np.dot(alpha_star_inv, temp.T)
        return mu_star, alpha_star_inv

    # Sample from Wishart distribution
    def sample_Wishart(self, W0, nu0):
        """
        Samples the precision matrix from a Wishart distribution.
        :param W0: The scale-matrix as a hyperparameter for the Wishart distribution. If no prior information,
                    it is recommended to set as an identity matrix
        :param nu0: The number of degrees of freedom as a hyperparameter for the Wishart distribution.
        :return: the new precision matrix sampled from the Wishart distribution
        """
        return wishart.rvs(nu0, W0, 1, self.rand_state)

    # Sample from Gaussian distribution
    def sample_Gaussian(self, mu, precision):
        """
        Samples from the multivariate Gaussian distribution.
        :param mu: The mean of the Gaussian distribution
        :param sigma: The precision of the Gaussian distribution.
        :return: returns the new sample (vector) from the multivariate Gaussian distribution
        """
        var = cholesky(precision)
        return (mu + np.dot(var, self.rand_state.randn(mu.shape[0], 1))).ravel()

    def build_rating_matrices(self, ratings, classes):
        """
        Build the rating matrices for each rank, with the set of all ratings observed that are currently assigned to class k.
        :param ratings: All the ratings
        :param classes: For each rating the classes (basically the rank), where they are assigned to.
        """
        self.ratings_csr_ = [build_user_item_matrix(self.n_user, self.n_item, ratings, classes, the_class) for the_class in range(self.K)]
        self.ratings_csc_ = [ratings_csr_.tocsc() for ratings_csr_ in self.ratings_csr_]

