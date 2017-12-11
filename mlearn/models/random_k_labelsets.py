import copy

import numpy as np
from joblib import Parallel, delayed

from ..utils import seed_random_state
from .model_wrapper import ModelWrapper

def train_single_clf(clf, X, y):
    lbl = np.packbits(y[:, clf[1]], axis=1)
    clf[0].fit(X, lbl.reshape(-1))
    return

class RandomKLabelsets():
    """
    References
    ----------
    .. [1] Tsoumakas, Grigorios, and Ioannis Vlahavas. "Random k-labelsets: An
           ensemble method for multilabel classification." Machine learning:
           ECML 2007 (2007): 406-417.
    """
    def __init__(self, base_clf, n_clfs, k, n_jobs=1, random_state=None):
        self.base_clf = base_clf
        self.clfs = None
        self.n_clfs = n_clfs
        self.K = k
        self.random_state_ = seed_random_state(random_state)
        self.n_jobs = n_jobs

    def train(self, X, y):
        self.n_labels = np.shape(y)[1]

        if self.clfs is None:
            self.clfs = []
            for i in range(self.n_clfs):
                clf = copy.deepcopy(self.base_clf)
                labelset = self.random_state_.choice(
                        range(self.n_labels), self.K, replace=False)
                self.clfs.append((SingleClassClfWrapper(clf), labelset))

        Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(train_single_clf)(self.clfs[i], X, y)
                for i in range(self.n_clfs)
            )

    def predict(self, X):
        votes = np.zeros((X.shape[0], self.n_labels))
        for clf, lbl in self.clfs:
            pred = clf.predict(X)
            res = np.unpackbits(
                    clf.predict(X).reshape((X.shape[0], 1)).astype(np.uint8), axis=1).astype(int)
            votes[:, lbl] += (res[:, :len(lbl)] * 2 - 1)
        return votes >= 0
