""" Test the models
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.sparse as ss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlearn.utils import load_data

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
        model = BinaryRelevance(LogisticRegression(
            multi_class="ovr", solver="liblinear", random_state=1126))
        model.train(self.X_train, self.Y_train)
        br_pred_train = model.predict(self.X_train)
        br_pred_test = model.predict(self.X_test)

        for i in range(np.shape(self.Y_train)[1]):
            clf = LogisticRegression(
                multi_class="ovr", solver="liblinear", random_state=1126)
            clf.fit(self.X_train, self.Y_train[:, i])

            assert_array_equal(clf.predict(self.X_train).astype(int),
                               br_pred_train[:, i])
            assert_array_equal(clf.predict(self.X_test).astype(int),
                               br_pred_test[:, i])

    def test_rakel(self):
        from mlearn.models import RandomKLabelsets
        clf = LogisticRegression(
            multi_class="ovr", solver="liblinear", random_state=1126)
        model = RandomKLabelsets(
            clf, n_clfs=10, k=3, n_jobs=1, random_state=1126)
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

    def test_classifier_chains(self):
        from mlearn.models import ClassifierChains
        clf = LogisticRegression(
            multi_class="ovr", solver="liblinear", random_state=1126)
        model = ClassifierChains(clf)
        model.train(self.X_train, self.Y_train)
        pred_train = model.predict(self.X_train)[:5]
        pred_test = model.predict(self.X_test)[:5]

        assert_array_equal(pred_train,
                           [[1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],])
        assert_array_equal(pred_test,
                           [[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0],])

    def test_pcc_f1(self):
        from mlearn.models import ProbabilisticClassifierChains
        clf = LogisticRegression(
            multi_class="ovr", solver="liblinear", random_state=1126)
        model = ProbabilisticClassifierChains(
            clf, cost='f1', n_samples=100, random_state=1126)
        model.train(self.X_train, self.Y_train)
        pred_train = model.predict(self.X_train)[:5]
        pred_test = model.predict(self.X_test)[:5]

        assert_array_equal(pred_train,
                           [[1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],])
        assert_array_equal(pred_test,
                           [[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],])

    def test_pcc_rank(self):
        from mlearn.models import ProbabilisticClassifierChains
        clf = LogisticRegression(
            multi_class="ovr", solver="liblinear", random_state=1126)
        model = ProbabilisticClassifierChains(
            clf, "rankloss", n_samples=100, random_state=1126)
        model.train(self.X_train, self.Y_train)
        pred_train = model.predict(self.X_train)[:5]
        pred_test = model.predict(self.X_test)[:5]

        assert_array_equal(pred_train,
                           [[1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 1, 1, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],])
        assert_array_equal(pred_test,
                           [[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0],])

    def test_csrpe(self):
        from mlearn.models import CSRPE
        from mlearn.criteria import pairwise_rank_loss
        clf = LogisticRegression(
            multi_class="ovr", solver="liblinear", random_state=1126)
        model = CSRPE(base_clf=clf, scoring_fn=pairwise_rank_loss,
                      n_clfs=100, n_jobs=-1, random_state=1126)
        model.train(self.X_train, self.Y_train)
        pred_train = model.predict(self.X_train)[:5]
        pred_test = model.predict(self.X_test)[:5]

        assert_array_equal(pred_train,
                           [[0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],])
        assert_array_equal(pred_test,
                           [[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],])

    def test_rethinkNet(self):
        from mlearn.models import RethinkNet
        from mlearn.criteria import sparse_pairwise_rank_loss
        n_features = self.X_train.shape[1]
        n_labels = self.Y_train.shape[1]
        scoring_fn = sparse_pairwise_rank_loss
        model = RethinkNet(
            n_features=n_features,
            n_labels=n_labels,
            scoring_fn=scoring_fn,
            b=3,
            random_state=1126,
        )
            
        model.train(self.X_train, self.Y_train)
        pred_train = model.predict(self.X_train)[:5]
        pred_test = model.predict(self.X_test)[:5]

        assert_array_equal(pred_train.toarray(),
                           [[0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1],])
        assert_array_equal(pred_test.toarray(),
                           [[0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0],])