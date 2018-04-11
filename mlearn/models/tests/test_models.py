""" Test the models
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.sparse as ss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlearn.utils import load_data
from mlearn.criteria import (
    reweighting,
    pairwise_hamming_loss,
    pairwise_rank_loss,
    pairwise_f1_score,
    pairwise_accuracy_score,
)

class ModelTestCase(unittest.TestCase):
    """
    Make sure the accelerated version corresponds to the naive implementation.
    """

    def setUp(self):
        X, Y = load_data('./examples/data/scene')
        self.X_train, self.X_test, self.Y_train, self.Y_test = \
            train_test_split(X, Y, test_size=0.3, random_state=1126)

    def test_binary_relevance(self):
        from mlearn.models import BinaryRelevance
        model = BinaryRelevance(LogisticRegression(random_state=1126))
        model.train(self.X_train, self.Y_train)
        br_pred_train = model.predict(self.X_train)
        br_pred_test = model.predict(self.X_test)

        for i in range(np.shape(self.Y_train)[1]):
            clf = LogisticRegression(random_state=1126)
            clf.fit(self.X_train, self.Y_train[:, i])

            assert_array_equal(clf.predict(self.X_train).astype(int),
                               br_pred_train[:, i])
            assert_array_equal(clf.predict(self.X_test).astype(int),
                               br_pred_test[:, i])

    def test_rakel(self):
        from mlearn.models import RandomKLabelsets
        model = RandomKLabelsets(LogisticRegression(random_state=1126),
                                 n_clfs=10, k=3, n_jobs=1, random_state=1126)
        model.train(self.X_train, self.Y_train)
        pred_train = model.predict(self.X_train)[:5]
        pred_test = model.predict(self.X_test)[:5]

        assert_array_equal(pred_train,
                           [[1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],])
        assert_array_equal(pred_test,
                           [[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],])