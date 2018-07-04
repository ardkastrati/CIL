import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

class SGDCollaborativeRegressor(BaseEstimator, RegressorMixin):

    def sample_with_dist(self, dist, k):
        rand = np.random.random()
        sample = min([np.sqrt(float(i + 1) / k) for i in range(5) if rand < dist[i]])
        return sample


    def initialize_with_training_distribution(self, y):
        count, _ = np.histogram(y, 5)
        fractions = count / len(y)
        fc = [sum(fractions[0:i + 1]) for i in range(5)]

        U_Z_training = np.ones((self.D + self.N, self.k))
        for row in range(self.D + self.N):
            for column in range(self.k):
                U_Z_training[row, column] = self.sample_with_dist(fc, self.k)

        return U_Z_training

    def gradient(self, U_Z, d, n, X_dn):

        U = U_Z[:self.D, :]
        Z = U_Z[self.D:, :]

        UZ_dn = np.matmul(U[d, :], Z[n, :].T)
        gradient_U = -(X_dn - UZ_dn) * Z[n, :] + 2*self.reg_factor*U[d,:]
        gradient_Z = -(X_dn - UZ_dn) * U[d, :] + 2*self.reg_factor*Z[n,:]

        return gradient_U, gradient_Z

    def stochastic_gradient_descent(self, X, y):

        U_Z = self.initialize_with_training_distribution(y=y)

        for i in range(self.num_of_samples):
            rand = np.random.randint(len(X))
            (d, n) = X[rand]
            score = y[rand]

            gradient_U, gradient_Z = self.gradient(U_Z=U_Z, d=d, n=n, X_dn=score)

            U_Z[d, :] = U_Z[d, :] - self.eta * gradient_U
            U_Z[n + self.D, :] = U_Z[n + self.D, :] - self.eta * gradient_Z

        return U_Z

    def normalize(self, X):
        X = 4/(1 + np.exp(-X + 3)) + 1
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
        This should fit regressor. All the "work" should be done here.
        """
        if verbose: print("Starting training...")
        U_Z = self.stochastic_gradient_descent(X=X, y=y)

        U = U_Z[:self.D, :]
        Z = U_Z[self.D:, :]
        self.X_predict = np.matmul(U, Z.T)
        self.X_predict = self.normalize(self.X_predict)
        if verbose: print("Finished training")
        return self


    def predict(self, X):
        try:
            getattr(self, "X_predict")
        except AttributeError:
            raise RuntimeError("You must train the regressor before predicting data!")

        return([self.X_predict[x] for x in X])


    def score(self, X, y=None):
        y_predict = self.predict(X)
        return -mean_squared_error(y, y_predict)
