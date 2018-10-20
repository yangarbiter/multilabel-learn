import numpy as np
from .reweight import (
    reweight_pairwise_hamming_loss,
    reweight_pairwise_accuracy_score,
    reweight_pairwise_f1_score,
    reweight_pairwise_rank_loss,
)
from .sparse_criteria import (
    sparse_pairwise_hamming_loss,
    sparse_pairwise_accuracy_score,
    sparse_pairwise_f1_score,
    sparse_pairwise_rank_loss,
)
from .sparse_reweight import (
    sparse_reweight_pairwise_hamming_loss,
    sparse_reweight_pairwise_accuracy_score,
    sparse_reweight_pairwise_f1_score,
    sparse_reweight_pairwise_rank_loss,
)

def reweighting(criterion, truth, predict, use_truth=False):
    """
    truth, predict should be same shape 2-d matrix

    Li, Chun-Liang, and Hsuan-Tien Lin. "Condensed filter tree for
    cost-sensitive multi-label classification." Proceedings of the 31st
    International Conference on Machine Learning (ICML-14). 2014. APA
    """
    K = truth.shape[1]
    ret = np.zeros((truth.shape[0], K, 2), dtype=float)
    s = criterion(truth, predict)

    for i in range(K):
        temp = predict.copy()

        temp[:, i] = 0
        s0 = criterion(truth, temp)
        ret[:, i, 0] = s0

        temp[:, i] = 1
        s1 = criterion(truth, temp)
        ret[:, i, 1] = s1

    return np.abs(ret[:, :, 0] - ret[:, :, 1])
    

def pairwise_hamming_loss(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    Z = Z.astype(int)
    Y = Y.astype(int)
    return (np.abs(Z - Y).astype(float)).mean(axis=1)

def pairwise_f1_score(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    # calculate F1 by sum(2*y_i*h_i) / (sum(y_i) + sum(h_i))
    Z = np.asarray(Z, dtype=int)
    Y = np.asarray(Y, dtype=int)
    up = 2 * np.sum(Z & Y, axis=1).astype(float)
    down1 = np.sum(Z, axis=1)
    down2 = np.sum(Y, axis=1)

    down = (down1 + down2)
    ind = (down==0) # avoid divide by zero error
    down[ind] = 1.
    up[ind] = 1.

    return up / down

def pairwise_rank_loss(Z, Y): #truth(Z), prediction(Y)
    """
    Z: truth
    Y: predict
    Z and Y should be the same size 2-d matrix
    """
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    return rankloss + tie0 + tie1

def pairwise_accuracy_score(Z, Y):
    f1 = 1.0 * ((Z>0) & (Y>0)).sum(axis=1)
    f2 = 1.0 * ((Z>0) | (Y>0)).sum(axis=1)
    f1[f2==0] = 1.0
    f1[f2>0] /= f2[f2>0]
    return f1