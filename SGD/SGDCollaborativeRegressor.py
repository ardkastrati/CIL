"""
Stochastic gradient descent for Collaborative Filtering
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class SGDCollaborativeRegressor(BaseEstimator, RegressorMixin):

    def sample_with_dist(self, dist, k):
        """
        Sample a random number between 0 and 1.
        :param dist: list of percentiles of each bin in the histogram of the data.
        :param k: rank of the matrix.
        :returns: minimum number between the percentile and a random number.
        """
        rand = np.random.random()
        sample = min([np.sqrt(float(i + 1) / k) for i in range(5) if rand < dist[i]])
        return sample


    def initialize_with_training_distribution(self, y):
        """
        # Initialize the user-rating matrix by sampling from the data distribution.
        :param y: list of ratings
        :returns: initialized user-rating matrix.
        """
        count, _ = np.histogram(y, 5)
        fractions = count / len(y)
        fc = [sum(fractions[0:i + 1]) for i in range(5)]

        U_Z_training = np.ones((self.D + self.N, self.k))
        for row in range(self.D + self.N):
            for column in range(self.k):
                U_Z_training[row, column] = self.sample_with_dist(fc, self.k)

        return U_Z_training

    def gradient(self, U_Z, d, n, X_dn):
        """
        Calculate the gradient of U and V matrices.
        :param U_Z: user-rating matrix of shape (n_user+n_rating, k).
        :param d: random number to get a random sample from the matrix.
        :param n: random number to get a random sample from the matrix.
        :param X_dn: rating of (d,n).
        :returns: gradient of U and gradient of Z.
        """
        U = U_Z[:self.D, :]
        Z = U_Z[self.D:, :]

        UZ_dn = np.matmul(U[d, :], Z[n, :].T)
        gradient_U = -(X_dn - UZ_dn) * Z[n, :] + 2*self.reg_factor*U[d,:]
        gradient_Z = -(X_dn - UZ_dn) * U[d, :] + 2*self.reg_factor*Z[n,:]

        return gradient_U, gradient_Z

    def stochastic_gradient_descent(self, X, y):
        """
        Perform stochastic gradient descent on the user-rating matrix.
        :param X: List of user-movie matrix. Each element is a tuple of format (user_id, movie_id)
        :param y: List of ratings.
        :returns: The updated user-rating matrix. It is a numpy array of shape (n_user+n_items, k)
        """
        U_Z = self.initialize_with_training_distribution(y=y)

        for i in range(self.num_of_samples):
            rand = np.random.randint(len(X))
            (d, n) = X[rand]
            score = y[rand]

            gradient_U, gradient_Z = self.gradient(U_Z=U_Z, d=d, n=n, X_dn=score)

            U_Z[d, :] = U_Z[d, :] - self.eta * gradient_U
            U_Z[n + self.D, :] = U_Z[n + self.D, :] - self.eta * gradient_Z

        return U_Z

    def clip(self, X):
        """
        Clip the ratings between 1 to 5.
        :param X: Predicted matrix. It is a numpy array of shape (n_users, n_items)
        :returns: Predicted matrix with clipped predictions from 1 to 5.
        """
        X[X>5] = 5
        X[X<1] = 1
        return X


    def __init__(self, k=1, eta=0.005, reg_factor = 1, num_of_samples = 100000, D=10000, N=1000):
        """
        Called when initializing the regressor
        """
        self.k = k
        self.eta = eta
        self.reg_factor = reg_factor
        self.num_of_samples = num_of_samples
        self.D = D
        self.N = N
        self.X_predict = []


    def fit(self, X, y=None, verbose=False):
        """
        Train the model using SGD.
        :param X: List of user-movie matrix. Each element is a tuple of format (user_id, movie_id)
        :param y: List of ratings.
        :param verbose: If True, prints running status.
        """
        if verbose: print("Starting training...")
        U_Z = self.stochastic_gradient_descent(X=X, y=y)

        U = U_Z[:self.D, :]
        Z = U_Z[self.D:, :]
        self.X_predict = np.matmul(U, Z.T)
        self.X_predict = self.clip(self.X_predict)
        if verbose: print("Finished training")
        return self

    def predict(self, X):
        """
        Predict target values of X.
        :param X: List of user-movie matrix to predict. Each element is a tuple of format (user_id, movie_id)
        :returns: numpy array
        """
        try:
            getattr(self, "X_predict")
        except AttributeError:
            raise RuntimeError("You must train the regressor before predicting data!")

        return([self.X_predict[x] for x in X])

    # MSE as score
    def score(self, X, y=None):
        """
        Calculate MSE score
        :param X: List of user-movie matrix to predict. Each element is a tuple of format (user_id, movie_id)
        :param y: List of real ratings from X.
        :return: MSE.
        """
        y_predict = self.predict(X)
        return -mean_squared_error(y, y_predict)
