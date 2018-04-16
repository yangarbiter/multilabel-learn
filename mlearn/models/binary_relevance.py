"""Binary Relevance Module
"""
import copy

import numpy as np

from .model_wrapper import ModelWrapper

class BinaryRelevance():
    """Binary Relevance

    Parameters
    ----------
    base_clf : sklearn classifier object instance.

    Attributes
    ----------
    clfs\_ : list
        A list of the model instances used in this algorithm.

    References
    ----------
    Tsoumakas, Grigorios, and Ioannis Katakis. "Multi-label classification:
    An overview." International Journal of Data Warehousing and Mining 3.3
    (2006).
    """

    def __init__(self, base_clf):
        self.base_clf = copy.copy(ModelWrapper(base_clf))
        self.n_labels = None
        self.clfs_ = []

    def train(self, X, y):
        self.n_labels = np.shape(y)[1]
        self.clfs_ = []

        for i in range(self.n_labels):
            clf = copy.deepcopy(self.base_clf)
            clf.fit(X, y[:, i])
            self.clfs_.append(clf)

    def predict(self, X):
        prediction = np.zeros((X.shape[0], self.n_labels))
        for i in range(self.n_labels):
            prediction[:, i] = self.clfs_[i].predict(X)
        return prediction
