# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3


import numpy as np
from cython.view cimport array as cvarray


cpdef double[:, ::1] invert1(double[:, ::1] xyz, double[::1] center):
    """
    Inversion operation on `xyz` with `center` as inversion center.
    :param xyz:    Nx3 coordinate array
    :param center: 1x3 coordinate array
    """
    cdef int i, n
    n = xyz.shape[0]
    got = np.empty((n, 3))
    for i in range(n):
        got[i, 0] = 2 * center[0] - xyz[i, 0]
        got[i, 1] = 2 * center[1] - xyz[i, 1]
        got[i, 2] = 2 * center[2] - xyz[i, 2]
    return got


cpdef double[:, ::1] invert2(double[:, ::1] xyz, double[::1] center):
    """
    Inversion operation on `xyz` with `center` as inversion center.

    :param xyz:    Nx3 coordinate array
    """
    cdef size_t i, n
    n = xyz.shape[0]
    # see https://stackoverflow.com/questions/18462785/what-is-the-recommended-way-of-allocating-memory-for-a-typed-memory-view
    got = cvarray(shape=(n, 3), itemsize=sizeof(double), format="d")
    cdef double[:, ::1] mv = got

    for i in range(n):
        mv[i, 0] = 2 * center[0] - xyz[i, 0]
        mv[i, 1] = 2 * center[1] - xyz[i, 1]
        mv[i, 2] = 2 * center[2] - xyz[i, 2]
    return mv
