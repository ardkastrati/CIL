"""
Reference papers: "Bayesian Probabilistic Matrix Factorization using MCMC"
                 R. Salakhutdinov and A.Mnih.
                 25th International Conference on Machine Learning (ICML-2008)
                Li, Dongsheng, et al. "Mixture-Rank Matrix Approximation for Collaborative Filtering.
                 "Advances in Neural Information Processing Systems. 2017.
Implementation of bpmf borrowed from: https://github.com/chyikwei/recommend
"""

import logging
from six.moves import xrange
import numpy as np
from numpy.linalg import inv, cholesky
from numpy.random import RandomState
from scipy.stats import wishart, norm
from surprise import Prediction

from utils.base import ModelBase
from utils.exceptions import NotFittedError
from utils.datasets import build_user_item_matrix
from utils.validation import check_ratings
from utils.evaluation import RMSE

logger = logging.getLogger(__name__)

class BPMRMF(ModelBase):
    """
    Bayesian Probabilistic Matrix Factorization with Mixture Ranking
    """
    def __init__(self, n_user, n_item, n_feature=[8], beta=2.0,
                 beta0_user=2.0, nu0_user=None, mu0_user=0.,
                 beta0_item=2.0, nu0_item=None, mu0_item=0.,
                 converge=1e-5, seed=None,
                 alpha=1.,
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
        self.converge = converge
        self.alpha = alpha #Dirichlet parameter

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

        # data state
        self.iter_ = 0
        self.mean_rating_ = None
        self.ratings_csr_ = None
        self.ratings_csc_ = None
        self.avg_ratings_ = np.zeros((self.n_user, self.n_item))
        self.MCMC_iteration = 0

    def MCMC(self):
        """
        Update predictions
        :param ratings:
        :param validation:
        :param test:
        :return:
        """

        iteration = self.MCMC_iteration

        # alpha * beta over all users and features
        class_probabilities = np.ndarray((self.K, self.n_user, self.n_item))

        for k in range(self.K):
            class_probabilities[k] = np.dot(self.user_class_probability_[k, :].reshape(self.n_user, 1), self.item_class_probability_[k, :].reshape(1, self.n_item))

        sum = class_probabilities.sum(axis=0)
        class_probabilities[:, sum == 0] = 0.5
        class_probabilities /= class_probabilities.sum(axis=0)
        # U * V^T weighted with alpha and beta
        R_hat = np.zeros((self.n_user, self.n_item), dtype='float')

        for k in range(self.K):
            curr_R_hat = np.dot(self.user_features_[k],  self.item_features_[k].T)
            R_hat += class_probabilities[k] * curr_R_hat

        R_hat += self.mean_rating_

        if self.max_rating:  # cut the prediction rate.
            R_hat[R_hat > self.max_rating] = self.max_rating

        if self.min_rating:
            R_hat[R_hat < self.min_rating] = self.min_rating

        self.avg_ratings_ *= (iteration / (iteration + 1))
        self.avg_ratings_ += R_hat / (iteration + 1)

        self.MCMC_iteration += 1


    def fit(self, ratings, validation, n_iters=50):

        # Check correctness of the ratings matrix
        check_ratings(ratings, self.n_user, self.n_item, self.max_rating, self.min_rating)

        # Initialize the classes randomly
        self.classes = self.rand_state.randint(0, self.K, size=len(ratings))
        self.build_rating_matrices(ratings, self.classes)

        self.mean_rating_ = np.mean(ratings[:, 2])

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
            self.MCMC()
            self.validate(ratings, validation)

            self.iter_ += 1

        return self

    def predict(self, data):
        if not self.mean_rating_:
            raise NotFittedError()

        preds = np.array([self.avg_ratings_[x[0], x[1]] for x in data])

        if self.max_rating:
            preds[preds > self.max_rating] = self.max_rating

        if self.min_rating:
            preds[preds < self.min_rating] = self.min_rating

        return preds

    """
    Just like in surprise library, so that can be easily run.
  
    def predict(self, uid, iid):
        if not self.mean_rating_:
            raise NotFittedError()

        est = self.estimate(int(uid), int(iid))
        return Prediction(uid, iid, 0.0, est, None)

    def estimate(self, user, item):
        est = self.avg_ratings_[user, item]

        #if self.max_rating and self.min_rating:  # cut the prediction rate.
         #   est = np.clip(est, 1.0, 5.0)
        return est
    """
    def validate(self,ratings, val):
        """
        Check RMSE in validation set
        :param val: Validation set, same format as training
        :return: None
        """
        train_preds = self.predict(ratings[:, :2])
        train_rmse = RMSE(train_preds, ratings[:, 2])
        print("iteration: {}, train RMSE: {}".format(self.iter_, train_rmse), end=" ")

        val_preds = self.predict(val[:, :2])
        val_rmse = RMSE(val_preds, val[:, 2])
        print("val RMSE: {}".format(val_rmse))

    def sample_user_class_probability(self):

        occurences = np.array([self.ratings_csr_[k].getnnz(axis=1) for k in range(self.K)], dtype=float)
        occurences += self.alpha/self.K
        occurences /= (occurences.sum(axis=0) + self.alpha)
        user_class_probabilities = np.array([self.rand_state.dirichlet(occurences[:, user_id]) for user_id in range(self.n_user)]).T
        return user_class_probabilities

    def sample_item_class_probability(self):
        occurences = np.array([self.ratings_csr_[k].getnnz(axis=0) for k in range(self.K)], dtype=float)
        occurences += self.alpha / self.K
        occurences /= (occurences.sum(axis=0) + self.alpha)
        item_class_probabilities = np.array([self.rand_state.dirichlet(occurences[:, item_id]) for item_id in range(self.n_item)]).T
        return item_class_probabilities

    def sample_user_item_classes(self, ratings):
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
        alpha_user = []
        mu_user = []
        for k in range(self.K):
            mu0_star, beta0_star, nu0_star, W0_star = \
                self.bayesian_update(self.n_user, self.mu0_user[k], self.beta0_user[k], self.nu0_user[k], self.W0_user[k], self.user_features_[k], self.n_feature[k])

            alpha_user.append(self.sample_Wishart(W0_star, nu0_star))
            mu_user.append(self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_user[k]))))
        return mu_user, alpha_user

    def sample_item_params(self):
        alpha_item = []
        mu_item = []
        for k in range(self.K):
            mu0_star, beta0_star, nu0_star, W0_star = \
                self.bayesian_update(self.n_item, self.mu0_item[k], self.beta0_item[k], self.nu0_item[k], self.W0_item[k], self.item_features_[k], self.n_feature[k])

            alpha_item.append(self.sample_Wishart(W0_star, nu0_star))
            mu_item.append(self.sample_Gaussian(mu0_star, inv(np.dot(beta0_star, alpha_item[k]))))
        return mu_item, alpha_item

    def sample_user_features(self):
        user_features_ = []
        for k in range(self.K):
            curr_user_features_ = np.zeros((self.n_user, self.n_feature[k]), dtype='float64')
            for user_id in xrange(self.n_user):
                indices = self.ratings_csr_[k][user_id, :].indices
                features = self.item_features_[k][indices, :]

                rating = self.ratings_csr_[k][user_id, :].data - self.mean_rating_
                rating = np.reshape(rating, (rating.shape[0], 1))

                mu_star, alpha_star_inv = self.conjugate_prior(self.mu_user[k], self.alpha_user[k], features, rating)
                curr_user_features_[user_id] = self.sample_Gaussian(mu_star, alpha_star_inv)

            user_features_.append(curr_user_features_)

        return user_features_

    def sample_item_features(self):
        item_features_ = []
        for k in range(self.K):
            curr_item_features_ = np.zeros((self.n_item, self.n_feature[k]), dtype='float64')

            for item_id in xrange(self.n_item):
                indices = self.ratings_csc_[k][:, item_id].indices
                features = self.user_features_[k][indices, :]

                rating = self.ratings_csc_[k][:, item_id].data - self.mean_rating_
                rating = np.reshape(rating, (rating.shape[0], 1))

                mu_star, alpha_star_inv = self.conjugate_prior(self.mu_item[k], self.alpha_item[k], features, rating)
                curr_item_features_[item_id] = self.sample_Gaussian(mu_star, alpha_star_inv)

            item_features_.append(curr_item_features_)

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

    def build_rating_matrices(self, ratings, classes):
        self.ratings_csr_ = [build_user_item_matrix(self.n_user, self.n_item, ratings, classes, the_class) for the_class in range(self.K)]
        self.ratings_csc_ = [ratings_csr_.tocsc() for ratings_csr_ in self.ratings_csr_]