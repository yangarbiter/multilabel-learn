"""Binary Relevance
"""
"""Binary Relevance Module
"""
import copy

import numpy as np

from .model_wrapper import DummyClf

class BinaryRelevance():
    def __init__(self, base_clf):
        self.base_clf = copy.copy(base_clf)

    def train(self, X, y):
        self.n_labels = np.shape(y)[1]
        self.clfs = []

        for i in range(self.n_labels):
            if len(np.unique(y[:, i])) == 1:
                clf = DummyClf()
            else:
                clf = copy.deepcopy(self.base_clf)
            clf.fit(X, y[:, i])
            self.clfs.append(clf)

    def predict(self, X):
        pred = np.zeros((X.shape[0], self.n_labels))
        for i in range(self.n_labels):
            pred[:, i] = self.clfs[i].predict(X)
        return pred
