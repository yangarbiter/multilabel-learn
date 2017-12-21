import numpy as np
import gzip
import pickle
import itertools as itt
import os
import errno

def pairwise_hamming(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    return -np.abs(Z - Y).mean(axis=1)


def pairwise_f1(Z, Y):
    """
    Z and Y should be the same size 2-d matrix
    """
    # calculate F1 by sum(2*y_i*h_i) / (sum(y_i) + sum(h_i))
    Z = Z.astype(int)
    Y = Y.astype(int)
    up = 2*np.sum(Z & Y, axis=1).astype(float)
    down1 = np.sum(Z, axis=1)
    down2 = np.sum(Y, axis=1)

    down = (down1 + down2)
    down[down==0] = 1.
    up[down==0] = 1.

    #return up / (down1 + down2)
    #assert np.all(up / (down1 + down2) == up/down) == True
    return up / down


def pairwise_rankloss(Z, Y): #truth(Z), prediction(Y)
    """
    Z and Y should be the same size 2-d matrix
    """
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    return -(rankloss + tie0 + tie1)


def pairwise_acc(Z, Y):
    f1 = 1.0 * ((Z>0) & (Y>0)).sum(axis=1)
    f2 = 1.0 * ((Z>0) | (Y>0)).sum(axis=1)
    f1[f2<=0] = 1.0
    f1[f2>0] /= f2[f2>0]
    return f1


def get_scoring_fn(scoring):
    if scoring == 'hamming':
        scoring_fn = pairwise_hamming
    elif scoring == 'f1':
        scoring_fn = pairwise_f1
    elif scoring == 'rankloss':
        scoring_fn = pairwise_rankloss
    elif scoring == 'acc':
        scoring_fn = pairwise_acc
    else:
        print("err:", [scoring])
    return scoring_fn

def seed_random_state(seed):
    """Turn seed into np.random.RandomState instance
    """
    if (seed is None) or (isinstance(seed, int)):
        return np.random.RandomState(seed)
    elif isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r can not be used to generate numpy.random.RandomState"
                     " instance" % seed)
