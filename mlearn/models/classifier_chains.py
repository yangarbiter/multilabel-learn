"""Classifier Chain module
"""
import copy

import numpy as np

from ..utils import get_scoring_fn
from .model_wrapper import ModelWrapper

class ClassifierChains():
    """
    Classifier Chain

    Parameters
    ----------
    base_clf : sklearn classifier object instance.

    Attributes
    ----------
    clfs\_ : list
        A list of the model instances used in this algorithm.

    References
    ----------
    .. [1] Read, Jesse, et al. "Classifier chains for multi-label
           classification." Machine learning 85.3 (2011): 333-359.
    """

    def __init__(self, base_clf):
        self.base_clf = ModelWrapper(copy.deepcopy(base_clf))
        self.n_labels = None
        self.clfs_ = []

    def train(self, X, y):
        self.n_labels = np.shape(y)[1]

        for i in range(self.n_labels):
            clf = copy.deepcopy(self.base_clf)

            pred = np.zeros((len(X), i))
            for j in range(i):
                tempX = np.hstack((X, pred[:, :j]))
                pred[:, j] = self.clfs_[j].predict(tempX)

            clf.fit(np.hstack((X, pred[:, :i])), y[:, i])

            self.clfs_.append(clf)

    def predict(self, X):
        pred = np.zeros((len(X), self.n_labels))
        for i in range(self.n_labels):
            tempX = np.hstack((X, pred[:, :i]))
            pred[:, i] = self.clfs_[i].predict(tempX)

        return pred.astype(int)