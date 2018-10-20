#cython: boundscheck=False

import numpy as np
import itertools as itt
from scipy import sparse
from joblib import parallel, delayed
import multiprocessing

import cython
from cython.view cimport array as cvarray
from cython.parallel cimport parallel, prange
cimport numpy as np

def pairwise_hamming(Z, y):
    #return iter_over_Z(Z, y)
    return -((Z-sparse.eye(np.shape(Z)[0], 1).dot(y)).mean(1)).toarray()
    #return -(Z-y).mean()

def pairwise_hamming_full(Z, Y):
    T = Z - Y
    T.data = np.abs(T.data)
    return -T.mean(1)

def pairwise_f1_full(Z, Y):
    sz = np.array(Z.sum(1))[:, 0]
    sy = np.array(Y.sum(1))[:, 0]

    up = np.array((Z.multiply(Y)).sum(1))[:, 0]

    down = (sz+sy)
    down[down==0] = 1.
    up[down==0] = 1.

    up = up.astype(float)
    return 2 * up/down

def pairwise_acc_full(Z, Y):
    sz = np.array(Z.sum(1))[:, 0]
    sy = np.array(Y.sum(1))[:, 0]

    up = np.array((Z.multiply(Y)).sum(1))[:, 0]

    down = (sz + sy - up)
    down[down==0] = 1.
    up[down==0] = 1.

    up = up.astype(np.float32)
    return up/down


def pairwise_f1(Z, y):
    sz = np.array(Z.sum(1))[:, 0]
    sy = np.array(y.sum(1))[:, 0]

    up = (Z.dot(y.transpose())).toarray()[:, 0]

    down = (sz+sy)
    down[down==0] = 1.
    up[down==0] = 1.

    up = up.astype(float)
    return 2 * up/down

def reweight_pairwise_hamming(truth, pred, use_true=False):
    cdef int i, K
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2))
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

        if use_true:
            now -= np.abs(truth[:, i] - pred[:, i])

    return -ret

def reweight_pairwise_rankloss(truth, pred, use_true=False):
    cdef int i, N
    cdef np.ndarray[np.int64_t, ndim=1] p01, p10, p11, p00, t, p
    cdef float r

    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2), np.float32)
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

    return -ret

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_reweight_pairwise_rankloss(truth, pred, use_true=False):
    cdef int i, N = pred.shape[0], K = pred.shape[1], j, n, k, tj, pj
    cdef int use_true_t = use_true
    cdef np.float32_t [:] up = np.zeros(N, np.float32)
    cdef np.float32_t [:] down = np.zeros(N, np.float32)
    cdef np.float32_t [:, :] ret
    cdef np.ndarray[np.float32_t, ndim=2] output = \
            np.zeros((pred.shape[0], pred.shape[1]), np.float32)
    #cdef np.ndarray[np.int32_t, ndim=1] tind, tptr, pind, pptr
    cdef np.int32_t [:] tind, tptr, pind, pptr
    cdef np.int32_t [:] ti, pi
    #cdef np.ndarray[np.int64_t, ndim=1] p01, p10, p11, p00, t, p
    cdef np.int32_t p01, p10, p11, p00

    truth.eliminate_zeros()
    pred.eliminate_zeros()
    truth.sort_indices()
    pred.sort_indices()

    tind = truth.indices.astype(np.int32)
    pind = pred.indices.astype(np.int32)
    tptr = truth.indptr.astype(np.int32)
    pptr = pred.indptr.astype(np.int32)

    ret = output

    with nogil:
        for i in range(N):
            ti = tind[tptr[i]: tptr[i+1]]
            pi = pind[pptr[i]: pptr[i+1]]

            p01 = p10 = p11 = p00 = 0
            tj = 0
            pj = 0

            for j in range(K):
                while tj < ti.shape[0] and ti[tj] < j:
                    tj += 1
                while pj < pi.shape[0] and pi[pj] < j:
                    pj += 1

                if ti[tj] == j and pi[pj] == j:
                    p11 += 1
                elif ti[tj] == j and pi[pj] != j:
                    p10 += 1
                elif ti[tj] != j and pi[pj] == j:
                    p01 += 1
                else:
                    p00 += 1

            r = p01 * p10 + 0.5 * p00 * p10 + 0.5 * p01 * p11

            for j in range(K):
                k = -1
                for k in range(pi.shape[0]):
                    if j == pi[k]:
                        break

                if j == pi[k]:
                    k = -1
                    for k in range(ti.shape[0]):
                        if j == ti[k]:
                            break
                    if j == ti[k]:
                        ret[i, j] = c_abs(r - (p01 * (p10 + 1)
                                             + 0.5 * p00 * (p10 + 1)
                                             + 0.5 * p01 * (p11 - 1)))
                    else:
                        ret[i, j] = c_abs(r - ((p01 - 1) * p10
                                             + 0.5 * (p00 + 1) * p10
                                             + 0.5 * (p01 - 1) * p11))
                else:
                    k = -1
                    for k in range(ti.shape[0]):
                        if j == ti[k]:
                            break
                    if j == ti[k]:
                        ret[i, j] = c_abs(r - (p01 * (p10 - 1)
                                             + 0.5 * p00 * (p10 - 1)
                                             + 0.5 * p01 * (p11 + 1)))
                    else:
                        ret[i, j] = c_abs(r - ((p01 + 1) * p10
                                             + 0.5 * (p00 - 1) * p10
                                             + 0.5 * (p01 + 1) * p11))

    return output

def p_reweight_pairwise_f1(
        np.ndarray[np.int8_t, ndim=2]truth, np.ndarray[np.int8_t, ndim=2] pred,
        use_true=False):
    cdef int i, N
    cdef float up, down

    ret = np.zeros((pred.shape[0], pred.shape[1], 2), np.float)
    N = pred.shape[0]

    for i in range(N):
        t = truth[i]
        p = pred[i]

        up = np.bitwise_and(t, p).sum()
        down = t.sum() + p.sum()

        # 1 -> 0
        ind = np.where(p == 1)[0]
        if down == 0:
            ret[i, ind, 1] = 1.
        else:
            ret[i, ind, 1] = 2 * up / down

        down -= 1
        if down == 0:
            ret[i, ind, 0] = 1.
        else:
            ii = np.where(t[ind] == 1)
            ret[i, ind[ii], 0] = 2 * (up - 1) / down
            ii = np.where(t[ind] == 0)
            ret[i, ind[ii], 0] = 2 * up / down

        down += 1

        # 0 -> 1
        ind = np.where(p == 0)[0]
        if down == 0:
            ret[i, ind, 0] = 1.
        else:
            ret[i, ind, 0] = 2 * up / down

        down += 1
        if down == 0:
            ret[i, ind, 1] = 1.
        else:
            ii = np.where(t[ind] == 1)
            ret[i, ind[ii], 1] = 2 * (up + 1) / down
            ii = np.where(t[ind] == 0)
            ret[i, ind[ii], 1] = 2 * up / down

    return ret

cdef double c_abs(double a) nogil:
    if a > 0: return a
    return -a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_reweight_pairwise_f1(truth, pred, use_true=False):
    cdef int i, N = pred.shape[0], K = pred.shape[1], j, n, k
    cdef int use_true_t = use_true
    cdef np.float32_t [:] up = np.zeros(N, np.float32)
    cdef np.float32_t [:] down = np.zeros(N, np.float32)
    cdef np.float32_t [:, :] ret
    cdef np.ndarray[np.float32_t, ndim=2] output = \
            np.zeros((pred.shape[0], pred.shape[1]), np.float32)
    #cdef np.ndarray[np.int32_t, ndim=1] tind, tptr, pind, pptr
    cdef np.int32_t [:] tind, tptr, pind, pptr
    cdef np.int32_t [:] ti, pi

    truth.eliminate_zeros()
    pred.eliminate_zeros()
    #truth.sort_indices()
    #pred.sort_indices()

    tind = truth.indices.astype(np.int32)
    pind = pred.indices.astype(np.int32)
    tptr = truth.indptr.astype(np.int32)
    pptr = pred.indptr.astype(np.int32)

    ret = output

    with nogil:
        for i in range(N):
            ti = tind[tptr[i]: tptr[i+1]]
            pi = pind[pptr[i]: pptr[i+1]]

            if use_true_t:
                down[i] = ti.shape[0] + ti.shape[0]
                up[i] = ti.shape[0]

                if down[i] == 0:
                    ret[i, :] = c_abs(.5 - up[i] / (down[i]+1))
                else:
                    ret[i, :] = c_abs(up[i] / down[i] - up[i] / (down[i]+1))

                # truth 1
                for j in range(ti.shape[0]):
                    # down[i] <= 1 won't happen
                    ret[i, ti[j]] = c_abs((up[i] - 1.) / (down[i] - 1.) - .5)
            else:
                for j in range(ti.shape[0]):
                    for k in range(pi.shape[0]):
                        if ti[j] == pi[k]:
                            up[i] += 1
                down[i] = ti.shape[0] + pi.shape[0]

                if down[i] == 0:
                    ret[i, :] = c_abs(.5 - up[i] / (down[i]+1))
                else:
                    ret[i, :] = c_abs(up[i] / down[i] - up[i] / (down[i]+1))

                # truth 1
                for j in range(ti.shape[0]):
                    k = -1
                    for k in range(pi.shape[0]):
                        # pred 1
                        if ti[j] == pi[k]:
                            if down[i] == 0:
                                ret[i, ti[j]] = 0.
                            elif down[i] == 1:
                                ret[i, ti[j]] = c_abs(up[i] / down[i] - .5)
                            else:
                                ret[i, ti[j]] = c_abs(up[i] / down[i] - (up[i]-1) / (down[i]-1))
                            break

                    if pi.shape[0] == 0 or (k == (pi.shape[0] - 1) and (ti[j] != pi[k])):
                        if down[i] == 0:
                            ret[i, ti[j]] = c_abs(0.5 - (up[i]+1) / (down[i]+1))
                        else:
                            ret[i, ti[j]] = c_abs(up[i] / down[i] - (up[i]+1) / (down[i]+1))

                # pred 1
                for j in range(pi.shape[0]):
                    k = -1
                    for k in range(ti.shape[0]):
                        if pi[j] == ti[k]:
                            break
                    # truth 0
                    if ti.shape[0] == 0 or ((k == (ti.shape[0] - 1)) and (ti[k] != pi[j])):
                        if down[i] == 1:
                            if ti.shape[0] > 0:
                                ret[i, ti[j]] = c_abs(up[i] / down[i] - .5)
                        elif down[i] == 0:
                            ret[i, pi[j]] = 0.
                        else:
                            ret[i, pi[j]] = c_abs(up[i] / down[i] - up[i] / (down[i]-1))

    return 2 * output


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_reweight_pairwise_acc(truth, pred, use_true=False):
    cdef int i, N = pred.shape[0], K = pred.shape[1], j, n, k
    cdef int use_true_t = use_true
    cdef np.float32_t [:] up = np.zeros(N, np.float32)
    cdef np.float32_t [:] down = np.zeros(N, np.float32)
    cdef np.float32_t [:, :] ret
    cdef np.ndarray[np.float32_t, ndim=2] output = \
            np.zeros((pred.shape[0], pred.shape[1]), np.float32)
    #cdef np.ndarray[np.int32_t, ndim=1] tind, tptr, pind, pptr
    cdef np.int32_t [:] tind, tptr, pind, pptr
    cdef np.int32_t [:] ti, pi

    truth.eliminate_zeros()
    pred.eliminate_zeros()
    #truth.sort_indices()
    #pred.sort_indices()

    tind = truth.indices.astype(np.int32)
    pind = pred.indices.astype(np.int32)
    tptr = truth.indptr.astype(np.int32)
    pptr = pred.indptr.astype(np.int32)

    ret = output

    with nogil:
        for i in range(N):
            ti = tind[tptr[i]: tptr[i+1]]
            pi = pind[pptr[i]: pptr[i+1]]

            if use_true_t:
                # TODO Not correct now
                down[i] = ti.shape[0] + ti.shape[0]
                up[i] = ti.shape[0]

                if down[i] == 0:
                    ret[i, :] = c_abs(.5 - up[i] / (down[i]+1))
                else:
                    ret[i, :] = c_abs(up[i] / down[i] - up[i] / (down[i]+1))

                # truth 1
                for j in range(ti.shape[0]):
                    # down[i] <= 1 won't happen
                    ret[i, ti[j]] = c_abs((up[i] - 1.) / (down[i] - 1.) - .5)
            else:
                down[i] = ti.shape[0] + pi.shape[0]
                for j in range(ti.shape[0]):
                    for k in range(pi.shape[0]):
                        if ti[j] == pi[k]:
                            up[i] += 1
                            down[i] -= 1

                # truth == 0, pred == 0
                if down[i] == 0:
                    ret[i, :] = c_abs(1. - up[i] / (down[i]+1))
                else:
                    ret[i, :] = c_abs(up[i] / down[i] - up[i] / (down[i]+1))

                # truth 1
                for j in range(ti.shape[0]):
                    k = -1
                    for k in range(pi.shape[0]):
                        # pred 1
                        if ti[j] == pi[k]:
                            if down[i] == 0:
                                # won't happen
                                ret[i, ti[j]] = 0.
                            elif down[i] == 1:
                                # won't happen
                                ret[i, ti[j]] = c_abs(up[i] / down[i] - 0.)
                            else:
                                ret[i, ti[j]] = c_abs(up[i] / down[i] - (up[i]-1) / (down[i]))
                            break

                    # pred 0
                    if pi.shape[0] == 0 or (k == (pi.shape[0] - 1) and (ti[j] != pi[k])):
                        if down[i] == 0:
                            # won't happen
                            ret[i, ti[j]] = c_abs(1.0 - (up[i]+1) / (down[i]))
                        else:
                            ret[i, ti[j]] = c_abs(up[i] / down[i] - (up[i]+1) / (down[i]))

                # pred 1
                for j in range(pi.shape[0]):
                    k = -1
                    for k in range(ti.shape[0]):
                        # truth 1 (duplicated)
                        if pi[j] == ti[k]:
                            break
                    # truth 0
                    if ti.shape[0] == 0 or ((k == (ti.shape[0] - 1)) and (ti[k] != pi[j])):
                        if down[i] == 0:
                            ret[i, pi[j]] = 0.
                        elif down[i] == 1:
                            ret[i, ti[j]] = c_abs(up[i] / down[i] - 1.)
                        else:
                            ret[i, pi[j]] = c_abs(up[i] / down[i] - up[i] / (down[i]-1))

    return output

def reweight_pairwise_f1(truth, pred, use_true=False):
    cdef int i, K
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2))
    K = np.shape(truth)[1]

    sz = truth.sum(1)
    sy = pred.sum(1)

    up = np.bitwise_and(truth, pred).sum(1)
    down = sz + sy

    for i in range(K):
        ret[:, i, 0] = 2 * up / down
        ret[:, i, 1] = 2 * up / down
        ret[np.where(down == 0), i, :] = 1.

    for i in range(K):
        # 1 -> 0
        ind = np.where(pred[:, i] == 1)[0]
        if len(ind) > 0:
            down[ind] -= 1
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==1)
            up[ind] -= c
            ret[ind, i, 0] = 2 * up[ind] / down[ind]
            ret[ind[np.where(down[ind]==0)], i, 0] = 1.

            down[ind] += 1
            up[ind] += c

        # 0 -> 1
        ind = np.where(pred[:, i] == 0)[0]
        if len(ind) > 0:
            down[ind] += 1
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
            up[ind] += c
            ret[ind, i, 1] = 2 * up[ind] / down[ind]
            ret[ind[np.where(down[ind]==0)], i, 1] = 1.

            down[ind] -= 1
            up[ind] -= c

        if use_true:
            ind = np.where(pred[:, i] == 0)[0]
            if len(ind) > 0:
                c1 = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
                up[ind] += c1
                down[ind] += c1

            ind = np.where(pred[:, i] == 1)[0]
            if len(ind) > 0:
                c0 = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==1)
                down[ind] -= c0

    return ret

def reweight_pairwise_acc(truth, pred):
    cdef int i, K
    truth = np.asarray(truth, int)
    pred = np.asarray(pred, int)

    ret = np.zeros((pred.shape[0], pred.shape[1], 2))
    K = np.shape(truth)[1] # n_labels

    up = np.bitwise_and(truth, pred).sum(1)
    down = np.bitwise_or(truth, pred).sum(1)

    for i in range(K):
        ret[:, i, 0] = up / down
        ret[:, i, 1] = up / down
        ret[np.where(down == 0), i, :] = 1. # in the case of divide zero

    for i in range(K):
        # 1 -> 0
        ind = np.where(pred[:, i] == 1)[0]
        if len(ind) > 0:
            # 1, 1
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==1)
            up[ind] -= c

            # 0, 1
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==1)
            down[ind] -= c

            ret[ind, i, 0] = up[ind] / down[ind]

            up[ind] += c
            down[ind] += c

        # 0 -> 1
        ind = np.where(pred[:, i] == 0)[0]
        if len(ind) > 0:
            # 1, 0
            c = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
            up[ind] += c

            # 0, 0
            c = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==0)

            down[ind] += c
            ret[ind, i, 1] = up[ind] / down[ind]

            up[ind] += c
            down[ind] -= c

        # TODO
        #if use_true:
        #    ind = np.where(pred[:, i] == 0)[0]
        #    if len(ind) > 0:
        #        c1 = np.bitwise_and(truth[ind, i]==1, pred[ind, i]==0)
        #        up[ind] += c1
        #        down[ind] += c1

        #    ind = np.where(pred[:, i] == 1)[0]
        #    if len(ind) > 0:
        #        c0 = np.bitwise_and(truth[ind, i]==0, pred[ind, i]==1)
        #        down[ind] -= c0

    return ret


def pairwise_rankloss_old(Z, Y):
    rankloss = ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie0 = 0.5 * ((Z==0) & (Y==0)).sum(axis=1) * ((Z==1) & (Y==0)).sum(axis=1)
    tie1 = 0.5 * ((Z==0) & (Y==1)).sum(axis=1) * ((Z==1) & (Y==1)).sum(axis=1)
    return -(rankloss + tie0 + tie1)

def pairwise_rankloss(Z, y):
    cdef list ret
    cdef np.ndarray[np.int32_t, ndim=1] zind, zptr, yind
    cdef int i, j, c, K, len_ztpr, len_yind
    cdef float l, n0, n1
    ret = []
    y.eliminate_zeros()
    Z.eliminate_zeros()
    yind = y.indices.astype(np.int32)
    yind = np.sort(yind)
    zptr = Z.indptr.astype(np.int32)
    Z.sort_indices()
    zind = Z.indices.astype(np.int32)
    K = np.shape(y)[1]

    len_zptr = len(zptr)
    len_yind = len(yind)

    for c in range(len_zptr-1):
        with nogil:
            n0 = 0.5 * (K - (zptr[c+1] - zptr[c]))
            n1 = 0.5 * (zptr[c+1] - zptr[c])
            l = 0.
            j = 0
            i = zptr[c]
            while i < zptr[c+1] and j < len_yind:
                #print i, j, zind[i], yind[j], l
                if zind[i] < yind[j]:
                    l += n0
                    i += 1
                elif zind[i] > yind[j]:
                    l += n1
                    j += 1
                else:
                    i += 1
                    j += 1

            if i < zptr[c+1]:
                l += (zptr[c+1] - i) * n0
            elif j < len_yind:
                l += (len_yind - j) * n1
        ret.append(l)
    return -np.array(ret)

def pairwise_rankloss_full(Z, Y):
    cdef list ret
    cdef np.ndarray[np.int32_t, ndim=1] zind, zptr, yind, yptr
    cdef int i, j, c, K
    cdef float l, n0, n1
    ret = []
    Y.eliminate_zeros()
    Y.sort_indices()
    Z.eliminate_zeros()
    Z.sort_indices()

    yptr = Y.indptr.astype(np.int32)
    yind = Y.indices.astype(np.int32)
    zptr = Z.indptr.astype(np.int32)
    zind = Z.indices.astype(np.int32)

    K = np.shape(Y)[1]
    for c in range(len(zptr)-1):
        n0 = 0.5 * (K - (zptr[c+1] - zptr[c]))
        n1 = 0.5 * (zptr[c+1] - zptr[c])
        l = 0.
        i = zptr[c]
        j = yptr[c]
        while i < zptr[c+1] and j < yptr[c+1]:
            if zind[i] < yind[j]:
                l += n0
                i += 1
            elif zind[i] > yind[j]:
                l += n1
                j += 1
            else:
                i += 1
                j += 1

        if i < zptr[c+1]:
            l += (zptr[c+1] - i) * n0
        elif j < yptr[c+1]:
            l += (yptr[c+1] - j) * n1
        ret.append(l)
    return -np.array(ret)


def pairwise_acc(Z, y):
    cdef list ret
    cdef np.ndarray[np.int32_t, ndim=1] zind, zptr, yind
    cdef int i, j, c, len_zptr, len_yind
    cdef float n0, n1
    ret = []
    y.eliminate_zeros()
    Z.eliminate_zeros()
    yind = y.indices.astype(np.int32)
    yind = np.sort(yind)
    zptr = Z.indptr.astype(np.int32)
    Z.sort_indices()
    zind = Z.indices.astype(np.int32)

    len_zptr = len(zptr)
    len_yind = len(yind)

    for c in range(len_zptr-1):
        with nogil:
            n0 = 0.
            n1 = 0.
            j = 0
            i = zptr[c]
            while i < zptr[c+1] and j < len_yind:
                if zind[i] < yind[j]:
                    n1 += 1
                    i += 1
                elif zind[i] > yind[j]:
                    n1 += 1
                    j += 1
                else:
                    n0 += 1
                    n1 += 1
                    i += 1
                    j += 1

        if n1 != 0:
            ret.append(n0/n1)
        else:
            ret.append(1.)
    return np.array(ret)
