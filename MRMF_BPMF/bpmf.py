"""
Reference paper: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.
                 25th International Conference on Machine Learning (ICML-2008)
Core bpmf implementation borrowed from: https://github.com/chyikwei/recommend
Added bias and implicit data to the core implementation
"""

import logging
import os
from six.moves import xrange
import numpy as np

from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart

from utils.base import ModelBase
from utils.exceptions import NotFittedError
from utils.datasets import build_user_item_matrix
from utils.validation import check_ratings
from utils.evaluation import RMSE
import pickle
import os.path

logger = logging.getLogger(__name__)

class BPMF(ModelBase):
    """
    Biased Bayesian Probabilistic Matrix Factorization
    """

    def __init__(self, n_user, n_item, n_feature, beta=2.0,
                 beta0_user=2.0, nu0_user=None, mu0_user=0.,
                 beta0_item=2.0, nu0_item=None, mu0_item=0.,
                 bias=True,
                 implicit=True,
                 beta0_a=2.0, nu0_a=None, mu0_a=0.,
                 beta0_b=2.0, nu0_b=None, mu0_b=0.,
                 beta0_implicit=2.0, nu0_implicit=None, mu0_implicit=0.,
                 converge=1e-5, seed=None, max_rating=None,
                 min_rating=None,
                 burn_in = 10):

        super(BPMF, self).__init__()

        #General data
        self.n_user = n_user # number of users
        self.n_item = n_item # number of movies
        self.n_feature = n_feature # rank of the matrix
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge
        self.bias = bias
        self.implicit = implicit
        self.burn_in = burn_in

        # Hyper Parameter
        self.beta = beta

        # Inv-Wishart (User features = U)
        self.W0_user = np.eye(n_feature, dtype='float64')
        self.beta0_user = beta0_user
        self.nu0_user = int(nu0_user) if nu0_user is not None else n_feature
        self.mu0_user = np.repeat(mu0_user, n_feature).reshape(n_feature, 1)  # a vector

        # Inv-Wishart (item features = V)
        self.W0_item = np.eye(n_feature, dtype='float64')
        self.beta0_item = beta0_item
        self.nu0_item = int(nu0_item) if nu0_item is not None else n_feature
        self.mu0_item = np.repeat(mu0_item, n_feature).reshape(n_feature, 1)

        # Inv-Wishart (User bias = a)
        self.W0_a = np.eye(1, dtype='float64')
        self.beta0_a = beta0_a
        self.nu0_a = int(nu0_a) if nu0_a is not None else 1
        self.mu0_a = np.repeat(mu0_a, 1).reshape(1, 1)  # a vector

        # Inv-Wishart (Item bias = b)
        self.W0_b = np.eye(1, dtype='float64')
        self.beta0_b = beta0_b
        self.nu0_b = int(nu0_b) if nu0_b is not None else 1
        self.mu0_b = np.repeat(mu0_b, 1).reshape(1, 1)  # b vector

        # Inv-Wishart (implicit item features = V)
        self.W0_implicit = np.eye(n_feature, dtype='float64')
        self.beta0_implicit = beta0_implicit
        self.nu0_implicit = int(nu0_implicit) if nu0_implicit is not None else n_feature
        self.mu0_implicit = np.repeat(mu0_implicit, n_feature).reshape(n_feature, 1)

        # Latent Variables
        self.mu_user = np.zeros((n_feature, 1), dtype='float64')
        self.mu_item = np.zeros((n_feature, 1), dtype='float64')

        self.alpha_user = np.eye(n_feature, dtype='float64')
        self.alpha_item = np.eye(n_feature, dtype='float64')

        # Bias
        self.mu_a = np.zeros((1, 1), dtype='float64')
        self.mu_b = np.zeros((1, 1), dtype='float64')

        self.alpha_a = np.eye(1, dtype='float64')
        self.alpha_b = np.eye(1, dtype='float64')

        #Implicit
        self.mu_implicit = np.zeros((n_feature, 1), dtype='float64')
        self.alpha_implicit = np.eye(n_feature, dtype='float64')

        self.initialize_features()

        # average user/item features and biases
        self.avg_user_features_ = np.zeros((n_user, n_feature))
        self.avg_item_features_ = np.zeros((n_item, n_feature))
        self.avg_aggregate_implicit_features_ = np.zeros((n_user, n_feature))
        self.avg_a = np.zeros((n_user, 1))
        self.avg_b = np.zeros((n_item, 1))

        # data state
        self.iter_ = 0
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

    def initialize_features(self):
        """
        Initialize the features randomly
        """
        # Nor particular reason of using 0.3 and 0.01
        self.user_features_ = 0.3 * self.rand_state.rand(self.n_user, self.n_feature)
        self.item_features_ = 0.3 * self.rand_state.rand(self.n_item, self.n_feature)
        self.implicit_features_ = 0.01 * self.rand_state.rand(self.n_item, self.n_feature)
        self.a_ = np.zeros((self.n_user, 1))
        self.b_ = np.zeros((self.n_item, 1))

        # Aggregation of the implicit data, makes the implementation faster
        self.aggregate_implicit_ = np.zeros((self.n_user, self.n_feature))
        # Store the normalization factor here
        self.normalization_implicit = np.zeros((self.n_user, 1))

    def MCMC(self, iteration):
        """
        :param iteration:
        :return:
        """
        self.avg_user_features_ *= (iteration / (iteration + 1.))
        self.avg_user_features_ += (self.user_features_ / (iteration + 1.))

        self.avg_item_features_ *= (iteration / (iteration + 1.))
        self.avg_item_features_ += (self.item_features_ / (iteration + 1.))

        if self.bias:
            self.avg_a *= (iteration / (iteration + 1.))
            self.avg_a += (self.a_ / (iteration + 1.))

            self.avg_b *= (iteration / (iteration + 1.))
            self.avg_b += (self.b_ / (iteration + 1.))

        if self.implicit:
            self.avg_aggregate_implicit_features_ *= (iteration / (iteration + 1.))
            self.avg_aggregate_implicit_features_ += (self.aggregate_implicit_ / (iteration + 1.))


    def fit(self, ratings, implicit_ratings, val, n_iters=50):
        """
        :param ratings: matrix of ratings with shape (n_sample, 3).
        Each row is (user_id, item_id, rating).
        :param implicit_ratings: implicit ratings from the test set
        :param val: validation set.
        :param n_iters: number of iterations
        :return: None
        """
        # Check correctness of the ratings matrix
        check_ratings(ratings, self.n_user, self.n_item, self.max_rating, self.min_rating)

        self.mean_rating_ = np.mean(ratings[:, 2])

        self.ratings_csr_ = build_user_item_matrix(self.n_user, self.n_item, ratings)
        self.ratings_csc_ = self.ratings_csr_.tocsc()

        self.implicit_ratings_csr_ = build_user_item_matrix(self.n_user, self.n_item, implicit_ratings)
        self.implicit_ratings_csc_ = self.ratings_csr_.tocsc()
        self.update_aggregate_implicit()


        last_rmse = None
        for iteration in xrange(n_iters):
            #Gibbs Sampling

            #Sampling hyperparameters
            if self.bias: self.mu_a, self.alpha_a = self.sample_user_bias_params()
            if self.bias: self.mu_b, self.alpha_b = self.sample_item_bias_params()
            if self.implicit: self.mu_implicit, self.alpha_implicit = self.sample_implicit_params()
            self.mu_item, self.alpha_item = self.sample_item_params()
            self.mu_user, self.alpha_user = self.sample_user_params()

            #Sampling parameters
            if self.bias: self.a_ = self.sample_user_bias()
            if self.bias: self.b_ = self.sample_item_bias()
            if self.implicit: self.implicit_features_ = self.sample_implicit_features()
            self.item_features_ = self.sample_item_features()
            self.user_features_ = self.sample_user_features()

            self.iter_ += 1
            if self.iter_ >= self.burn_in:
                # Do MCMC
                self.MCMC(self.iter_ - self.burn_in)

                # Check if no more iterations needed
                converged, last_rmse = self.check_convergence(ratings, last_rmse)

                #Validate current RMSE
                self.validate(val)
                if converged:
                    logger.info('Converged at iteration %d. stop.', self.iter_)
                    break
            else:
                print("Burn in iteration ", self.iter_)
        return self

    def predict(self, data):
        if not self.mean_rating_:
            raise NotFittedError()

        u_features = self.avg_user_features_.take(data.take(0, axis=1), axis=0)
        i_features = self.avg_item_features_.take(data.take(1, axis=1), axis=0)

        preds = np.sum(u_features * i_features, 1) + self.mean_rating_

        if self.bias:
            a = self.avg_a.take(data.take(0, axis=1), axis=0)
            b = self.avg_b.take(data.take(1, axis=1), axis=0)
            preds += a.ravel() + b.ravel()

        if self.implicit:
            aggregate_implicit_features = self.avg_aggregate_implicit_features_.take(data.take(0, axis=1), axis=0)
            preds += (u_features*aggregate_implicit_features).sum(-1)

        if self.max_rating:  # cut the prediction rate. 
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    def check_convergence(self, ratings, last_rmse):
        """
        Checks if the training error drop is below a threshold
        :param ratings: Matrix of ratings
        :param last_rmse: Last root mean squared error obtained
        :return: None
        """
        train_preds = self.predict(ratings[:, :2])
        train_rmse = RMSE(train_preds, ratings[:, 2])
        # print("iteration: {}, train RMSE: {}".format(self.iter_, train_rmse), end=" ")

        if last_rmse and (abs(train_rmse - last_rmse) < self.converge):
            logger.info('converges at iteration %d. stop.', self.iter_)
            return True, train_rmse
        else:
            return False, train_rmse

    def validate(self, val):
        """
        Test RMSE in the validation set
        :param val:
        :return:
        """
        val_preds = self.predict(val[:, :2])
        val_rmse = RMSE(val_preds, val[:, 2])
        self.val_scores.append(val_rmse)
        # print("val RMSE: {}".format(val_rmse))

    def sample_user_bias_params(self):
        mu0_star, beta0_star, nu0_star, W0_star = \
            self.bayesian_update(self.n_user, self.mu0_a, self.beta0_a, self.nu0_a, self.W0_a, self.a_, 1)
        alpha_a = self.sample_Wishart(W0_star, nu0_star)
        mu_a = self.sample_Gaussian(mu0_star, np.array([[1/(beta0_star*alpha_a)]]))
        return mu_a, alpha_a

    def sample_item_bias_params(self):
        mu0_star, beta0_star, nu0_star, W0_star = \
            self.bayesian_update(self.n_item, self.mu0_b, self.beta0_b, self.nu0_b, self.W0_b, self.b_, 1)

        alpha_b = self.sample_Wishart(W0_star, nu0_star)
        mu_b = self.sample_Gaussian(mu0_star, np.array([[1/(beta0_star*alpha_b)]]))
        return mu_b, alpha_b

    def sample_user_params(self):
        mu0_star, beta0_star, nu0_star, W0_star = \
            self.bayesian_update(self.n_user, self.mu0_user, self.beta0_user, self.nu0_user, self.W0_user, self.user_features_, self.n_feature)

        alpha_user = self.sample_Wishart(W0_star, nu0_star)
        mu_user = self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_user)))
        return mu_user, alpha_user

    def sample_item_params(self):
        mu0_star, beta0_star, nu0_star, W0_star = \
            self.bayesian_update(self.n_item, self.mu0_item, self.beta0_item, self.nu0_item, self.W0_item, self.item_features_, self.n_feature)

        alpha_item = self.sample_Wishart(W0_star, nu0_star)
        mu_item = self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_item)))
        return mu_item, alpha_item

    def sample_implicit_params(self):
        mu0_star, beta0_star, nu0_star, W0_star = \
            self.bayesian_update(self.n_item, self.mu0_implicit, self.beta0_implicit, self.nu0_implicit, self.W0_implicit,
                                 self.implicit_features_, self.n_feature)

        alpha_item = self.sample_Wishart(W0_star, nu0_star)
        mu_item = self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_item)))
        return mu_item, alpha_item

    def sample_user_bias(self):
        a_ = np.zeros((self.n_user, 1), dtype='float64')
        for user_id in xrange(self.n_user):
            indices = self.ratings_csr_[user_id, :].indices
            features = np.array([np.ones(len(indices))]).T

            rating = self.ratings_csr_[user_id, :].data \
                            - self.mean_rating_ \
                            - self.b_[indices].T  \
                            - np.dot(self.user_features_[user_id, :], ((self.item_features_[indices, :] + self.aggregate_implicit_[user_id, :]).T))
            rating = np.reshape(rating, (rating.shape[1], 1))
            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_a, self.alpha_a, features, rating)
            a_[user_id] = self.sample_Gaussian(mu_star, alpha_star_inv)
        return a_

    def sample_item_bias(self):
        b_ = np.zeros((self.n_item, 1), dtype='float64')

        for item_id in xrange(self.n_item):
            indices = self.ratings_csc_[:, item_id].indices
            features = np.array([np.ones(len(indices))]).T

            rating = self.ratings_csc_[:, item_id].data.T \
                     - self.mean_rating_ \
                     - self.a_[indices].T \
                     - (self.user_features_[indices, :] * (self.item_features_[item_id, :] + self.aggregate_implicit_[indices, :])).sum(-1)

            rating = np.reshape(rating, (rating.shape[1], 1))

            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_b, self.alpha_b, features, rating)
            b_[item_id] = self.sample_Gaussian(mu_star, alpha_star_inv)
        return b_

    def sample_user_features(self):
        user_features_ = np.zeros((self.n_user, self.n_feature), dtype='float64')

        for user_id in xrange(self.n_user):
            indices = self.ratings_csr_[user_id, :].indices
            features = self.item_features_[indices, :] + self.aggregate_implicit_[user_id, :]

            rating = self.ratings_csr_[user_id, :].data \
                     - self.mean_rating_ \
                     - self.a_[user_id] \
                     - self.b_[indices].T

            rating = np.reshape(rating, (rating.shape[1], 1))

            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_user, self.alpha_user, features, rating)
            user_features_[user_id] = self.sample_Gaussian(mu_star, alpha_star_inv)
        return user_features_

    def sample_item_features(self):
        item_features_ = np.zeros((self.n_item, self.n_feature), dtype='float64')

        for item_id in xrange(self.n_item):
            indices = self.ratings_csc_[:, item_id].indices
            features = self.user_features_[indices, :]

            rating = self.ratings_csc_[:, item_id].data \
                     - self.mean_rating_ \
                     - self.a_[indices].T \
                     - self.b_[item_id] \
                     - (self.aggregate_implicit_[indices, :] * self.user_features_[indices, :]).sum(-1)

            rating = np.reshape(rating, (rating.shape[1], 1))

            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_item, self.alpha_item, features, rating)
            item_features_[item_id] = self.sample_Gaussian(mu_star, alpha_star_inv)
        return item_features_

    def sample_implicit_features(self):
        implicit_features_ = np.zeros((self.n_item, self.n_feature), dtype='float64')

        for item_id in xrange(self.n_item):
            indices = self.ratings_csc_[:, item_id].indices
            features = self.user_features_[indices, :]

            rating = self.ratings_csc_[:, item_id].data \
                     - self.mean_rating_ \
                     - self.a_[indices].T \
                     - self.b_[item_id] \
                     - ((self.item_features_[item_id, :] - self.normalization_implicit[indices]*self.implicit_features_[item_id, :] + self.aggregate_implicit_[indices, :])
                        * self.user_features_[indices,:]).sum(-1)

            rating = np.reshape(rating, (rating.shape[1], 1))

            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_item, self.alpha_item, features, rating)
            implicit_features_[item_id] = self.sample_Gaussian(mu_star, alpha_star_inv)

        self.update_aggregate_implicit()
        return implicit_features_

    def bayesian_update(self, N, mu0, beta0, nu0, W0, evidence, evidence_n_feature):
        X_bar = np.mean(evidence, 0).reshape((evidence_n_feature, 1))
        S_bar = np.cov(evidence.T)
        diff_X_bar = mu0 - X_bar
        W0_star = inv(inv(W0) +  N * S_bar + np.dot(diff_X_bar, diff_X_bar.T) * (N * beta0) / (beta0 + N))

        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmertic here
        W0_star = (W0_star + W0_star.T) / 2.0

        nu0_star = nu0 + N
        beta0_star = beta0 + N
        mu0_star = (beta0 * mu0 + N*X_bar) / (beta0 + N)

        return mu0_star, beta0_star, nu0_star, W0_star

    def conjugate_prior(self, mu, alpha, features, rating):
        alpha_star = alpha + self.beta * np.dot(features.T, features)
        alpha_star_inv = inv(alpha_star)
        temp = (self.beta * np.dot(features.T, rating).T + np.dot(alpha, mu))
        mu_star = np.dot(alpha_star_inv, temp.T)
        return mu_star, alpha_star_inv

    def sample_Wishart(self, W0, nu0):
        return wishart.rvs(nu0, W0, 1, self.rand_state)

    def sample_Gaussian(self, mu, sigma):
        var = cholesky(sigma)
        return (mu + np.dot(var, self.rand_state.randn(mu.shape[0], 1))).ravel()

    def update_aggregate_implicit(self):
        for user_id in xrange(self.n_user):
            indices = self.implicit_ratings_csr_[user_id, :].indices
            features = self.implicit_features_[indices, :]
            self.normalization_implicit[user_id] = 1./len(indices)
            self.aggregate_implicit_[user_id] = 1./len(indices) * np.sum(features, axis=0)
