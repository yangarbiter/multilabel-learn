import copy

import numpy as np
from joblib import Parallel, delayed

from ..utils import seed_random_state
from .model_wrapper import ModelWrapper

def train_single_clf(clf, X, y):
    """
    helper function to train RAKEL in parallel
    """
    lbl = np.packbits(y[:, clf[1]], axis=1)
    clf[0].fit(X, lbl.reshape(-1))
    return

class RandomKLabelsets():
    """
    RAndom K labELsets (RAKEL)

    Parameters
    ----------
    base_clf: sklearn classifier object instance.

    n_clfs: int
        The total number of classifiers on different labelset to be trained.

    k: int
        The size of the labelset.

    n_jobs:  int, optional (default=1)
        The number of jobs to run in parallel for both fit and predict. If
        -1, then the number of jobs is set to the number of cores.
        
    random_state : {int, np.random.RandomState instance, None}, optional (default=None)
        If int or None, random_state is passed as parameter to generate
        np.random.RandomState instance. if np.random.RandomState instance,
        random_state is the random number generate.

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
        self.n_labels = None

    def train(self, X, y):
        self.n_labels = np.shape(y)[1]

        if self.clfs is None:
            self.clfs = []
            for _ in range(self.n_clfs):
                clf = copy.deepcopy(self.base_clf)
                labelset = self.random_state_.choice(
                    range(self.n_labels), self.K, replace=False)
                self.clfs.append((ModelWrapper(clf), labelset))

        Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(train_single_clf)(self.clfs[i], X, y)
            for i in range(self.n_clfs)
        )

    def predict(self, X):
        votes = np.zeros((X.shape[0], self.n_labels))
        for clf, lbl in self.clfs:
            pred = clf.predict(X)
            res = np.unpackbits(
                pred.reshape((X.shape[0], 1)).astype(np.uint8),
                axis=1).astype(int)
            votes[:, lbl] += (res[:, :len(lbl)] * 2 - 1)
        return (votes >= 0).astype(int)
