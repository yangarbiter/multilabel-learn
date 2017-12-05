
import copy

import numpy as np

from .utils import get_scoring_fn

class DummyClf():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.cls = int(y[0]) # 1 or 0

    def predict(self, X):
        return self.cls * np.ones(len(X))

    def predict_proba(self, X):
        ret = np.zeros((len(X), 2))
        ret[:, self.cls] = 1
        return ret

class BinaryRelevance():
    def __init__(self, base_clf, scoring_fn):
        self.base_clf = copy.copy(base_clf)
        self.scoring_fn = scoring_fn # for score

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

if __name__ == '__main__':
    main()
