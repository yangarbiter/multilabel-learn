""" Test the evaluation criteria
"""
import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import scipy.sparse as ss

from mlearn.criteria import (
    reweighting,
    pairwise_hamming_loss,
    pairwise_rank_loss,
    pairwise_f1_score,
    pairwise_accuracy_score,
)
from mlearn.criteria.sparse_criteria import (
    sparse_pairwise_hamming_loss,
    sparse_pairwise_rank_loss,
    sparse_pairwise_f1_score,
    sparse_pairwise_accuracy_score,
)
from mlearn.criteria.reweight import (
    reweight_pairwise_hamming_loss,
    reweight_pairwise_rank_loss,
    reweight_pairwise_f1_score,
    reweight_pairwise_accuracy_score,
)
from mlearn.criteria.sparse_reweight import (
    sparse_reweight_pairwise_hamming_loss,
    sparse_reweight_pairwise_rank_loss,
    sparse_reweight_pairwise_f1_score,
    sparse_reweight_pairwise_accuracy_score,
)


class CalcCriteriaTestCase(unittest.TestCase):
    """
    Make sure the accelerated version corresponds to the naive implementation.
    """

    def setUp(self):
        self.Z = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.int)

        self.Y = np.array([
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
        ], dtype=np.int)

    def test_hamming_loss(self):
        assert_array_equal(
            pairwise_hamming_loss(self.Z, self.Y),
            sparse_pairwise_hamming_loss(ss.csr_matrix(self.Z), ss.csr_matrix(self.Y))
        )

    def test_rank_loss(self):
        assert_array_equal(
            pairwise_rank_loss(self.Z, self.Y),
            sparse_pairwise_rank_loss(ss.csr_matrix(self.Z), ss.csr_matrix(self.Y))
        )

    def test_f1_score(self):
        assert_array_equal(
            pairwise_f1_score(self.Z, self.Y),
            sparse_pairwise_f1_score(ss.csr_matrix(self.Z), ss.csr_matrix(self.Y))
        )

    def test_accuracy_score(self):
        assert_array_equal(
            pairwise_accuracy_score(self.Z, self.Y),
            sparse_pairwise_accuracy_score(ss.csr_matrix(self.Z), ss.csr_matrix(self.Y))
        )

class ReweightingTestCase(unittest.TestCase):
    """
    Make sure the accelerated version corresponds to the naive implementation.
    """

    def setUp(self):
        self.Z = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],

            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int)

        self.Y = np.array([
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],

            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int)

    def test_hamming_loss(self):
        assert_array_almost_equal(
            reweighting(pairwise_hamming_loss, self.Z, self.Y, use_truth=False),
            reweight_pairwise_hamming_loss(self.Z, self.Y, use_truth=False),
        )
        # TODO
        #assert_array_almost_equal(
        #    reweighting(pairwise_hamming_loss, self.Z, self.Y, use_truth=False),
        #    sparse_reweight_pairwise_hamming_loss(
        #        ss.csr_matrix(self.Z), ss.csr_matrix(self.Y), use_truth=False),
        #)

    def test_rank_loss(self):
        assert_array_almost_equal(
            reweighting(pairwise_rank_loss, self.Z, self.Y, use_truth=False),
            reweight_pairwise_rank_loss(self.Z, self.Y, use_truth=False),
        )
        assert_array_almost_equal(
            reweighting(pairwise_rank_loss, self.Z, self.Y, use_truth=False),
            sparse_reweight_pairwise_rank_loss(
                ss.csr_matrix(self.Z), ss.csr_matrix(self.Y), use_truth=False),
        )

    def test_f1_score(self):
        assert_array_almost_equal(
            reweighting(pairwise_f1_score, self.Z, self.Y, use_truth=False),
            reweight_pairwise_f1_score(self.Z, self.Y, use_truth=False),
        )
        assert_array_almost_equal(
            reweighting(pairwise_f1_score, self.Z, self.Y, use_truth=False),
            sparse_reweight_pairwise_f1_score(
                ss.csr_matrix(self.Z), ss.csr_matrix(self.Y), use_truth=False),
        )

    def test_accuracy_score(self):
        assert_array_almost_equal(
            reweighting(pairwise_accuracy_score, self.Z, self.Y, use_truth=False),
            reweight_pairwise_accuracy_score(self.Z, self.Y, use_truth=False),
        )
        assert_array_almost_equal(
            reweighting(pairwise_accuracy_score, self.Z, self.Y, use_truth=False),
            sparse_reweight_pairwise_accuracy_score(
                ss.csr_matrix(self.Z), ss.csr_matrix(self.Y), use_truth=False),
        )


if __name__ == '__main__':
    unittest.main()