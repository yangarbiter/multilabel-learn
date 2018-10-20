# cython: language_level=3

import cython
import numpy as np

cimport numpy as np

cdef double c_abs(double a) nogil:
    if a > 0: return a
    return -a


def sparse_reweight_pairwise_hamming_loss(truth, pred, use_truth=False):
    # TODO
    pass

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_reweight_pairwise_rank_loss(truth, pred, use_truth=False):
    cdef int i, N = pred.shape[0], K = pred.shape[1], j, k, tj, pj
    cdef int use_truth_t = use_truth
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

                if (tj < ti.shape[0]) and (pj < pi.shape[0]) and ti[tj] == j and pi[pj] == j:
                    p11 += 1
                elif (tj < ti.shape[0] and ti[tj] == j):
                    p10 += 1
                elif (pj < pi.shape[0] and pi[pj] == j):
                    p01 += 1
                else:
                    p00 += 1

            r = p01 * p10 + 0.5 * p00 * p10 + 0.5 * p01 * p11

            for j in range(K):
                k = -1
                for k in range(pi.shape[0]):
                    if j == pi[k]:
                        break

                # pred = 1
                if k != -1 and j == pi[k]:
                    k = -1
                    for k in range(ti.shape[0]):
                        if j == ti[k]:
                            break
                    if k != -1 and j == ti[k]: # 1 1 -> 1 0
                        ret[i, j] = c_abs(r - (p01 * (p10 + 1)
                                                + 0.5 * p00 * (p10 + 1)
                                                + 0.5 * p01 * (p11 - 1)))
                    else: # 0 1 -> 0 0
                        ret[i, j] = c_abs(r - ((p01 - 1) * p10
                                                + 0.5 * (p00 + 1) * p10
                                                + 0.5 * (p01 - 1) * p11))
                # pred = 0
                else:
                    k = -1
                    for k in range(ti.shape[0]):
                        if j == ti[k]:
                            break
                    if k != -1 and j == ti[k]: # 1 0 -> 1 1
                        ret[i, j] = c_abs(r - (p01 * (p10 - 1)
                                                + 0.5 * p00 * (p10 - 1)
                                                + 0.5 * p01 * (p11 + 1)))
                    else: # 0 0 -> 0 1
                        ret[i, j] = c_abs(r - ((p01 + 1) * p10
                                                + 0.5 * (p00 - 1) * p10
                                                + 0.5 * (p01 + 1) * p11))

    return output

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_reweight_pairwise_f1_score(truth, pred, use_truth=False):
    cdef int i, N = pred.shape[0], K = pred.shape[1], j, n, k
    cdef int use_truth_t = use_truth
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

            if use_truth_t:
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
                            ret[i, pi[j]] = c_abs(up[i] / down[i] - .5)
                        elif down[i] == 0:
                            ret[i, pi[j]] = 0.
                        else:
                            ret[i, pi[j]] = c_abs(up[i] / down[i] - up[i] / (down[i]-1))

    return 2 * output


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def sparse_reweight_pairwise_accuracy_score(truth, pred, use_truth=False):
    cdef int i, N = pred.shape[0], K = pred.shape[1], j, n, k
    cdef int use_truth_t = use_truth
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

            if use_truth_t:
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
                                ret[i, ti[j]] = -1
                            else:
                                ret[i, ti[j]] = c_abs(up[i] / down[i] - (up[i]-1) / (down[i]))
                            break

                    # pred 0
                    if (pi.shape[0] == 0) or (k == (pi.shape[0] - 1) and (ti[j] != pi[k])):
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
                    if (ti.shape[0] == 0) or ((k == (ti.shape[0] - 1)) and (ti[k] != pi[j])):
                        if down[i] == 0:
                            # should not happend
                            ret[i, pi[j]] = -1
                        elif down[i] == 1:
                            ret[i, pi[j]] = c_abs(up[i] / down[i] - 1.)
                        else:
                            ret[i, pi[j]] = c_abs(up[i] / down[i] - up[i] / (down[i]-1))

    return output