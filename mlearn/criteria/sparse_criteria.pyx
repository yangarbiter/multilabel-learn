#cython: boundscheck=False

import cython
import numpy as np
from scipy import sparse

cimport numpy as np

def sparse_pairwise_hamming_loss(Z, Y):
    T = Z - Y
    T.data = np.abs(T.data)
    return np.array(T.mean(1), dtype=np.float).flatten()

def sparse_pairwise_rank_loss(Z, Y):
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
    return np.array(ret, dtype=np.float)

def sparse_pairwise_f1_score(Z, Y):
    sz = np.array(Z.sum(1), dtype=int)[:, 0]
    sy = np.array(Y.sum(1), dtype=int)[:, 0]

    up = np.array((Z.multiply(Y)).sum(1))[:, 0]

    down = (sz + sy)

    #### make 0, 0 -> 1
    ind = (down==0)
    down[ind] = 2. 
    up[ind] = 1
    ####

    up = up.astype(np.float)
    return 2 * up / down

def sparse_pairwise_accuracy_score(Z, Y):
    sz = np.array(Z.sum(1))[:, 0]
    sy = np.array(Y.sum(1))[:, 0]

    up = np.array((Z.multiply(Y)).sum(1))[:, 0]

    down = (sz + sy - up)
    up[down==0] = 1.
    down[down==0] = 1.

    up = up.astype(np.float)
    return up/down

#def sparse_pairwise_rank_loss(Z, y):
#    cdef list ret
#    cdef np.ndarray[np.int32_t, ndim=1] zind, zptr, yind
#    cdef int i, j, c, K, len_ztpr, len_yind
#    cdef float l, n0, n1
#    ret = []
#    y.eliminate_zeros()
#    Z.eliminate_zeros()
#    yind = y.indices.astype(np.int32)
#    yind = np.sort(yind)
#
#    zptr = Z.indptr.astype(np.int32)
#    Z.sort_indices()
#    zind = Z.indices.astype(np.int32)
#    K = np.shape(y)[1]
#
#    len_zptr = len(zptr)
#    len_yind = len(yind)
#
#    for c in range(len_zptr-1):
#        with nogil:
#            n0 = 0.5 * (K - (zptr[c+1] - zptr[c]))
#            n1 = 0.5 * (zptr[c+1] - zptr[c])
#            l = 0.
#            j = 0
#            i = zptr[c]
#            while i < zptr[c+1] and j < len_yind:
#                #print i, j, zind[i], yind[j], l
#                if zind[i] < yind[j]:
#                    l += n0
#                    i += 1
#                elif zind[i] > yind[j]:
#                    l += n1
#                    j += 1
#                else:
#                    i += 1
#                    j += 1
#
#            if i < zptr[c+1]:
#                l += (zptr[c+1] - i) * n0
#            elif j < len_yind:
#                l += (len_yind - j) * n1
#        ret.append(l)
#    return -np.array(ret)
#
#def sparse_pairwise_accuracy_score(Z, y):
#    cdef list ret
#    cdef np.ndarray[np.int32_t, ndim=1] zind, zptr, yind
#    cdef int i, j, c, len_zptr, len_yind
#    cdef float n0, n1
#    ret = []
#    y.eliminate_zeros()
#    Z.eliminate_zeros()
#    yind = y.indices.astype(np.int32)
#    yind = np.sort(yind)
#    zptr = Z.indptr.astype(np.int32)
#    Z.sort_indices()
#    zind = Z.indices.astype(np.int32)
#
#    len_zptr = len(zptr)
#    len_yind = len(yind)
#
#    for c in range(len_zptr-1):
#        with nogil:
#            n0 = 0.
#            n1 = 0.
#            j = 0
#            i = zptr[c]
#            while i < zptr[c+1] and j < len_yind:
#                if zind[i] < yind[j]:
#                    n1 += 1
#                    i += 1
#                elif zind[i] > yind[j]:
#                    n1 += 1
#                    j += 1
#                else:
#                    n0 += 1
#                    n1 += 1
#                    i += 1
#                    j += 1
#
#        if n1 != 0:
#            ret.append(n0/n1)
#        else:
#            ret.append(1.)
#    return np.array(ret)