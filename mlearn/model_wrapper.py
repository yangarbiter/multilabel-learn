
import numpy as np

class DummyClf():
    """This classifier handles training sets with only 0s or 1s to unify the
    interface.

    """

    def __init__(self):
        self.classes_ = [0, 1]

    def fit(self, X, y, sample_weight=None):
        self.cls = int(y[0]) # 1 or 0

    def train(self, dataset):
        _, y = zip(*dataset.get_labeled_entries())
        self.cls = int(y[0])

    def predict(self, X):
        return self.cls * np.ones(len(X))

    def predict_proba(self, X):
        ret = np.zeros((len(X), 2))
        ret[:, self.cls] = 1.
        return ret

class ClfWrapper():
    def __init__(self, *args, **kwargs):
        self._model = self.model_class(*args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        self._model.fit(X, y, *args, **kwargs)

    def train(self, X, y, *args, **kwargs):
        self._model.fit(X, y, *args, **kwargs)

    def predict(self, X, *args, **kwargs):
        self._model.predict(X, *args, **kwargs)

    def predict_proba(self, X, *args, **kwargs):
        self._model.predict_proba(X, *args, **kwargs)

class ModelWrapper():
    """This classifier handles training sets with only 0s or 1s to unify the
    interface.
    """
    def __init__(self, model):
        self.model = model
        self.cls = None

    def fit(self, X, y, sample_weight=None):
        if len(np.unique(y)) == 1:
            self.cls = int(y[0]) # 1 or 0
        else:
            self.model.fit(X, y)

    def train(self, X, y):
        if len(np.unique(y)) == 1:
            self.cls = int(y[0]) # 1 or 0
        else:
            self.model.train(X, y)

    def predict(self, X):
        if self.cls is not None:
            return self.cls * np.ones(len(X))
        else:
            return self.model.predict(X)
