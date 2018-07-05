"""
Base class of recommendation System
Core implementation borrowed from: https://github.com/chyikwei/recommend
"""

from abc import ABCMeta, abstractmethod


class ModelBase(object):

    """base class of recommendations"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def fit(self, train, validation, test_data, n_iters):
        """training models"""

    @abstractmethod
    def predict(self):
        """save model"""
