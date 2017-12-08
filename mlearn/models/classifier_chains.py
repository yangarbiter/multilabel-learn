
import gzip, copy

import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from six.moves import cPickle as pickle
from sklearn.tree import DecisionTreeClassifier

from ..utils import get_scoring_fn
from .model_wrapper import ModelWrapper

class ClassifierChains():
    """
    References
    ----------
    .. [1] Read, Jesse, et al. "Classifier chains for multi-label
           classification." Machine learning 85.3 (2011): 333-359.
    """

    def __init__(self, base_clf):
        self.base_clf = SingleClassClfWrapper(copy.deepcopy(base_clf))

    def train(self, X, y):
        self.n_labels = np.shape(y)[1]
        self.clfs = []

        for i in range(self.n_labels):
            clf = copy.deepcopy(self.base_clf)

            pred = np.zeros((len(X), i))
            for j in range(i):
                temp_X = np.hstack((X, pred[:, :j]))
                pred[:, j] = self.clfs[j].predict(temp_X)

            clf.fit(np.hstack((X, pred[:, :i])), y[:, i])

            self.clfs.append(clf)

    def predict(self, X):
        pred = np.zeros((len(X), self.n_labels))
        for i in range(self.n_labels):
            temp_X = np.hstack((X, pred[:, :i]))
            pred[:, i] = self.clfs[i].predict(temp_X)

        return pred
