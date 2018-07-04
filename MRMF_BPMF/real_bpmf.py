"""
Reference paper: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.
                 25th International Conference on Machine Learning (ICML-2008)
Core bpmf implementation borrowed from: https://github.com/chyikwei/recommend. Only
restructured so that is more easily readable.
"""

import logging
from six.moves import xrange
import numpy as np

from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart
from surprise import Prediction

from utils.base import ModelBase
from utils.exceptions import NotFittedError
from utils.datasets import build_user_item_matrix
from utils.validation import check_ratings
from utils.evaluation import RMSE
logger = logging.getLogger(__name__)

class BPMF(ModelBase):
    """
    Biased Bayesian Probabilistic Matrix Factorization
    """

    def __init__(self, n_user, n_item, n_feature, beta=2.0,
                 beta0_user=2.0, nu0_user=None, mu0_user=0.,
                 beta0_item=2.0, nu0_item=None, mu0_item=0.,
                 converge=1e-5, seed=None, max_rating=None,
                 min_rating=None,
                 burn_in = 0):

        super(BPMF, self).__init__()

        #General data
        self.n_user = n_user # number of users
        self.n_item = n_item # number of movies
        self.n_feature = n_feature # rank of the matrix
        self.rand_state = RandomState(seed)
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.converge = converge
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

        # Latent Variables
        self.mu_user = np.zeros((n_feature, 1), dtype='float64')
        self.mu_item = np.zeros((n_feature, 1), dtype='float64')

        self.alpha_user = np.eye(n_feature, dtype='float64')
        self.alpha_item = np.eye(n_feature, dtype='float64')

        self.initialize_features()

        # average user/item features
        self.avg_ratings_ = np.zeros((n_user, n_item))

        # data state
        self.iter_ = 0
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None

    def initialize_features(self):
        """
        Initialize the features randomly
        """
        # No particular reason of using 0.3
        self.user_features_ = 0.3 * self.rand_state.rand(self.n_user, self.n_feature)
        self.item_features_ = 0.3 * self.rand_state.rand(self.n_item, self.n_feature)

    def MCMC(self, iteration):
        """
        :param iteration:
        :return:
        """
        self.avg_ratings_ *= (iteration / (iteration + 1.))
        self.avg_ratings_ += (np.dot(self.user_features_, self.item_features_.T) / (iteration + 1.))


    def fit(self, ratings, val, n_iters=50):
        """
        :param ratings: matrix of ratings with shape (n_sample, 3).
        Each row is (user_id, item_id, rating).
        :param n_iters: number of iterations
        :return: None
        """
        # Check correctness of the ratings matrix
        check_ratings(ratings, self.n_user, self.n_item, self.max_rating, self.min_rating)

        self.mean_rating_ = np.mean(ratings[:, 2])

        self.ratings_csr_ = build_user_item_matrix(self.n_user, self.n_item, ratings)
        self.ratings_csc_ = self.ratings_csr_.tocsc()

        last_rmse = None
        for iteration in xrange(n_iters):
            #Gibbs Sampling

            #Sampling hyperparameters
            self.mu_item, self.alpha_item = self.sample_item_params()
            self.mu_user, self.alpha_user = self.sample_user_params()

            #Sampling parameters
            self.item_features_ = self.sample_item_features()
            self.user_features_ = self.sample_user_features()

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

            self.iter_ += 1
        return self


    def predict_all(self, data):
        if not self.mean_rating_:
            raise NotFittedError()

        preds = np.array([self.avg_ratings_[x[0],x[1]] for x in data]) + self.mean_rating_

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds


    """
      Just like in surprise library, so that can be easily run.
    """
    def predict(self, uid, iid):
        if not self.mean_rating_:
            raise NotFittedError()

        est = self.estimate(int(uid), int(iid))
        return Prediction(uid, iid, 0.0, est, None)

    def estimate(self, user, item):
        est = self.avg_ratings_[user, item]
        if self.max_rating and self.min_rating:  # cut the prediction rate.
            est = np.clip(est, 1.0, 5.0)
        return est

    def check_convergence(self, ratings, last_rmse):
        """
        Checks if the training error drop is below a threshold
        :param ratings: Matrix of ratings
        :param last_rmse: Last root mean squared error obtained
        :return: None
        """
        train_preds = self.predict_all(ratings[:, :2])
        train_rmse = RMSE(train_preds, ratings[:, 2])
        print("iteration: {}, train RMSE: {}".format(self.iter_, train_rmse), end=" ")

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
        val_preds = self.predict_all(val[:, :2])
        val_rmse = RMSE(val_preds, val[:, 2])
        print("val RMSE: {}".format(val_rmse))

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


    def sample_user_features(self):
        user_features_ = np.zeros((self.n_user, self.n_feature), dtype='float64')

        for user_id in xrange(self.n_user):
            indices = self.ratings_csr_[user_id, :].indices
            features = self.item_features_[indices, :]

            rating = self.ratings_csr_[user_id, :].data - self.mean_rating_

            rating = np.reshape(rating, (rating.shape[0], 1))

            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_user, self.alpha_user, features, rating)
            user_features_[user_id] = self.sample_Gaussian(mu_star, alpha_star_inv)
        return user_features_

    def sample_item_features(self):
        item_features_ = np.zeros((self.n_item, self.n_feature), dtype='float64')

        for item_id in xrange(self.n_item):
            indices = self.ratings_csc_[:, item_id].indices
            features = self.user_features_[indices, :]

            rating = self.ratings_csc_[:, item_id].data - self.mean_rating_

            rating = np.reshape(rating, (rating.shape[0], 1))

            mu_star, alpha_star_inv = self.conjugate_prior(self.mu_item, self.alpha_item, features, rating)
            item_features_[item_id] = self.sample_Gaussian(mu_star, alpha_star_inv)
        return item_features_


    def bayesian_update(self, N, mu0, beta0, nu0, W0, evidence, evidence_n_feature):
        X_bar = np.mean(evidence, 0).reshape((evidence_n_feature, 1))
        S_bar = np.cov(evidence.T)
        diff_X_bar = mu0 - X_bar
        W0_star = inv(inv(W0) +  N * S_bar + np.dot(diff_X_bar, diff_X_bar.T) * (N * beta0) / (beta0 + N))

        # Note: WI_post and WI_post.T should be the same.
        #       Just make sure it is symmetric here
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
