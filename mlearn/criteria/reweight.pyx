import cython
import numpy as np

cimport numpy as np

def reweight_pairwise_hamming_loss(truth, pred, use_truth=False):
    cdef int i, K
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2), dtype=np.float32)
    K = np.shape(truth)[1]

    now = np.abs(truth - pred).sum(axis=1)
    for i in range(K):
        ind = np.where(pred[:, i] == 1)[0]
        ret[ind, i, 1] = now[ind] / K
        ind = np.where(np.bitwise_and(truth[:, i]==0, pred[:, i]==1))[0]
        ret[ind, i, 0] = (now[ind] - 1) / K
        ind = np.where(np.bitwise_and(truth[:, i]==1, pred[:, i]==1))[0]
        ret[ind, i, 0] = (now[ind] + 1) / K

        ind = np.where(pred[:, i] == 0)[0]
        ret[ind, i, 0] = now[ind] / K
        ind = np.where(np.bitwise_and(truth[:, i]==0, pred[:, i]==0))[0]
        ret[ind, i, 1] = (now[ind] + 1) / K
        ind = np.where(np.bitwise_and(truth[:, i]==1, pred[:, i]==0))[0]
        ret[ind, i, 1] = (now[ind] - 1) / K

        if use_truth:
            now -= np.abs(truth[:, i] - pred[:, i])

    return np.abs(ret[:, :, 0] - ret[:, :, 1])

def reweight_pairwise_f1_score(truth, pred, use_truth=False):
    cdef int i, K
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2), dtype=np.float32)
    K = np.shape(truth)[1]

    sz = truth.sum(1)
    sy = pred.sum(1)

    up = np.bitwise_and(truth, pred).sum(1)
    down = sz + sy

    for i in range(K):
        idx = np.where(down!=0)[0]
        ret[idx, i, 0] = 2 * up[idx] / down[idx]
        ret[idx, i, 1] = 2 * up[idx] / down[idx]
        ret[np.where(down == 0), i, :] = 1.

    for i in range(K):
        # 1 -> 0
        ind = np.where(pred[:, i] == 1)[0]
        if len(ind) > 0:
            down[ind] -= 1
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==1)
            up[ind] -= c
            #ret[ind, i, 0] = 2 * up[ind] / down[ind]
            idx = np.where(down[ind]!=0)[0]
            ret[ind[idx], i, 0] = 2 * up[ind][idx] / down[ind][idx]
            ret[ind[np.where(down[ind]==0)], i, 0] = 1.

            down[ind] += 1
            up[ind] += c

        # 0 -> 1
        ind = np.where(pred[:, i] == 0)[0]
        if len(ind) > 0:
            down[ind] += 1
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
            up[ind] += c
            idx = np.where(down[ind]!=0)[0]
            ret[ind[idx], i, 1] = 2 * up[ind][idx] / down[ind][idx]
            #ret[ind[np.where(down[ind]==0)], i, 1] = 1.

            down[ind] -= 1
            up[ind] -= c

        if use_truth:
            ind = np.where(pred[:, i] == 0)[0]
            if len(ind) > 0:
                c1 = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
                up[ind] += c1
                down[ind] += c1

            ind = np.where(pred[:, i] == 1)[0]
            if len(ind) > 0:
                c0 = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==1)
                down[ind] -= c0

    return np.abs(ret[:, :, 0] - ret[:, :, 1])

def reweight_pairwise_rank_loss(truth, pred, use_truth=False):
    cdef int i, N
    cdef np.ndarray[np.int64_t, ndim=1] p01, p10, p11, p00, t, p
    cdef float r

    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2), dtype=np.float32)
    N = pred.shape[0]

    p01 = ((truth==0) & (pred==1)).sum(axis=1)
    p10 = ((truth==1) & (pred==0)).sum(axis=1)
    p00 = ((truth==0) & (pred==0)).sum(axis=1)
    p11 = ((truth==1) & (pred==1)).sum(axis=1)

    for i in range(N):
        t = truth[i]
        p = pred[i]

        r = p01[i] * p10[i] + 0.5 * p00[i] * p10[i] + 0.5 * p01[i] * p11[i]

        # 1 -> 0
        ind = np.where(p == 1)[0]
        ret[i, ind, 1] = r

        ii = np.where(t[ind] == 1)
        ret[i, ind[ii], 0] = p01[i] * (p10[i] + 1) \
                             + 0.5 * p00[i] * (p10[i] + 1) \
                             + 0.5 * p01[i] * (p11[i] - 1)
        ii = np.where(t[ind] == 0)
        ret[i, ind[ii], 0] = (p01[i] - 1) * p10[i] \
                             + 0.5 * (p00[i] + 1) * p10[i] \
                             + 0.5 * (p01[i] - 1) * p11[i]

        # 0 -> 1
        ind = np.where(p == 0)[0]
        ret[i, ind, 0] = r

        ii = np.where(t[ind] == 1)
        ret[i, ind[ii], 1] = p01[i] * (p10[i] - 1) \
                             + 0.5 * p00[i] * (p10[i] - 1) \
                             + 0.5 * p01[i] * (p11[i] + 1)
        ii = np.where(t[ind] == 0)
        ret[i, ind[ii], 1] = (p01[i] + 1) * p10[i] \
                             + 0.5 * (p00[i] - 1) * p10[i] \
                             + 0.5 * (p01[i] + 1) * p11[i]

    return np.abs(ret[:, :, 0] - ret[:, :, 1])

def reweight_pairwise_accuracy_score(truth, pred, use_truth=False):
    cdef int i, K
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2), dtype=np.float32)
    K = np.shape(truth)[1] # n_labels

    up = np.bitwise_and(truth, pred).sum(1)
    down = np.bitwise_or(truth, pred).sum(1)

    for i in range(K):
        idx = np.where(down != 0)[0]
        ret[idx, i, 0] = up[idx] / down[idx]
        ret[idx, i, 1] = up[idx] / down[idx]
        ret[np.where(down == 0), i, :] = 1. # in the case of divide zero

    for i in range(K):
        # 1 -> 0
        ind = np.where(pred[:, i] == 1)[0]
        if len(ind) > 0:
            # 1, 1
            c1 = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==1)
            up[ind] -= c1

            # 0, 1
            c2 = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==1)
            down[ind] -= c2

            idx = np.where(down[ind]!=0)[0]
            ret[ind[idx], i, 0] = up[ind[idx]] / down[ind[idx]]
            ret[ind[np.where(down[ind]==0)], i, 0] = 1.
            #ret[ind, i, 0] = up[ind] / down[ind]

            up[ind] += c1
            down[ind] += c2

        # 0 -> 1
        ind = np.where(pred[:, i] == 0)[0]
        if len(ind) > 0:
            # 1, 0
            c1 = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
            up[ind] += c1

            # 0, 0
            c2 = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==0)
            down[ind] += c2

            idx = np.where(down[ind]!=0)[0]
            ret[ind[idx], i, 1] = up[ind[idx]] / down[ind[idx]]
            ret[ind[np.where(down[ind]==0)], i, 1] = 1.
            #ret[ind, i, 1] = up[ind] / down[ind]

            up[ind] -= c1
            down[ind] -= c2

        # TODO
        #if use_truth:
        #    ind = np.where(pred[:, i] == 0)[0]
        #    if len(ind) > 0:
        #        c1 = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
        #        up[ind] += c1
        #        down[ind] += c1

        #    ind = np.where(pred[:, i] == 1)[0]
        #    if len(ind) > 0:
        #        c0 = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==1)
        #        down[ind] -= c0

    return np.abs(ret[:, :, 0] - ret[:, :, 1])

#def p_reweight_pairwise_f1(
#        np.ndarray[np.int8_t, ndim=2]truth, np.ndarray[np.int8_t, ndim=2] pred,
#        use_truth=False):
#    cdef int i, N
#    cdef float up, down
#
#    ret = np.zeros((pred.shape[0], pred.shape[1], 2), np.float)
#    N = pred.shape[0]
#
#    for i in range(N):
#        t = truth[i]
#        p = pred[i]
#
#        up = np.bitwise_and(t, p).sum()
#        down = t.sum() + p.sum()
#
#        # 1 -> 0
#        ind = np.where(p == 1)[0]
#        if down == 0:
#            ret[i, ind, 1] = 1.
#        else:
#            ret[i, ind, 1] = 2 * up / down
#
#        down -= 1
#        if down == 0:
#            ret[i, ind, 0] = 1.
#        else:
#            ii = np.where(t[ind] == 1)
#            ret[i, ind[ii], 0] = 2 * (up - 1) / down
#            ii = np.where(t[ind] == 0)
#            ret[i, ind[ii], 0] = 2 * up / down
#
#        down += 1
#
#        # 0 -> 1
#        ind = np.where(p == 0)[0]
#        if down == 0:
#            ret[i, ind, 0] = 1.
#        else:
#            ret[i, ind, 0] = 2 * up / down
#
#        down += 1
#        if down == 0:
#            ret[i, ind, 1] = 1.
#        else:
#            ii = np.where(t[ind] == 1)
#            ret[i, ind[ii], 1] = 2 * (up + 1) / down
#            ii = np.where(t[ind] == 0)
#            ret[i, ind[ii], 1] = 2 * up / down
#
#    return ret