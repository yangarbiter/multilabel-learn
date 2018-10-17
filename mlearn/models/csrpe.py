"""Cost-Sensitive Random Pair Encoding
"""
import copy

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from .model_wrapper import DummyClf
from ..utils import seed_random_state


class CSRPE():
    """ Cost-Sensitive Random Pair Encoding (CSRPE)

    References
    ----------
    .. [1] Yang, Yao-Yuan, et al. "Cost-Sensitive Reference Pair Encoding for
           Multi-Label Learning." arXiv preprint arXiv:1611.09461 (2016).
    """
    def __init__(self, scoring_fn, base_clf, n_clfs, n_jobs=1,
            random_state=None):
        self.scoring_fn = scoring_fn
        self.base_clf = base_clf
        self.nn = NearestNeighbors(1, algorithm='ball_tree', metric='euclidean',
                n_jobs=n_jobs)
        self.n_jobs = n_jobs
        self.n_clfs = n_clfs
        self.random_state_ = seed_random_state(random_state)

        self.n_labels = None
        self.clfs = None

    def encode(self, X):
        encoded = np.zeros((X.shape[0], len(self.clfs)))
        for i, clf in enumerate(self.clfs):
            encoded[:, i] = clf.predict(X)
        return encoded

    def train(self, X, y):
        if self.n_labels is None:
            self.n_labels = np.shape(y)[1]
            self.clfs = [CLF(self.base_clf, self.scoring_fn,
                rep_label=self.random_state_.randint(0, 2, (2, self.n_labels)))
                for i in range(self.n_clfs)]

        #self.tokens = np.vstack({tuple(r) for r in y})
        self.tokens = y

        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(train_single_clf)(self.clfs[i], X, y)
            for i in range(self.n_clfs)
        )

        self.nn.fit(self.encode(X))

    def predict(self, X):
        encoded = self.encode(X)
        ind = self.nn.kneighbors(encoded, 1, return_distance=False)
        ind = ind.reshape(-1)
        return self.tokens[ind]

    def predict_real(self, X):
        return self.predict_dist(X)

class CLF():
    """
    dummy classifier interface to run CSRPE in parallel
    """
    def __init__(self, base_clf, scoring_fn, rep_label=None, random_state=None):
        self.base_clf_ = copy.copy(base_clf)
        self.base_clf = None
        self.scoring_fn = scoring_fn
        self.random_state_ = seed_random_state(random_state)
        self.rep_label = rep_label

    def train(self, X, y):
        self.n_samples = np.shape(X)[0]
        self.n_labels = np.shape(y)[1]

        if self.rep_label is None:
            self.rep_label = self.random_state_.randint(0, 2,
                    (2, self.n_labels))

        score0 = self.scoring_fn(y, np.tile(self.rep_label[0], (self.n_samples, 1)))
        score1 = self.scoring_fn(y, np.tile(self.rep_label[1], (self.n_samples, 1)))
        lbl = (((score1 - score0) > 0) + 0.0)
        weight = np.abs(score1 - score0)
        if weight.sum() == 0:
            weight = np.ones_like(weight)
        else:
            weight /= weight.sum()
            weight *= weight.shape[0]

        if len(np.unique(lbl)) == 1:
            self.base_clf = DummyClf()
        else:
            self.base_clf = self.base_clf_
        self.base_clf.fit(X, lbl, sample_weight=weight)

    def predict(self, X):
        return self.base_clf.predict(X)

def train_single_clf(clf, X, y):
    clf.train(X, y)
    return
