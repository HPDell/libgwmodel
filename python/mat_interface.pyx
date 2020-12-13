cimport cbase
from cbase cimport GwmMatInterface
from libc.stdlib cimport malloc, free
import numpy as np

cdef class MatInterface:
    cdef GwmMatInterface _c_instance

    def __cinit__(self, rows, cols, double[:, ::1] array):
        self._c_instance = GwmMatInterface(rows, cols, &array[0, 0])

    def __dealloc__(self):
        cbase.gwmodel_delete_mat(&self._c_instance)


def numpy2mat(double[:, ::1] array):
    cdef unsigned long long rows = array.shape[0]
    cdef unsigned long long cols = array.shape[1]
    return MatInterface(rows, cols, array)


cdef mat2numpy(GwmMatInterface interface):
    cdef const double* src = interface.data
    cdef unsigned long long rows = interface.rows
    cdef unsigned long long cols = interface.cols
    cdef double[:, ::1] dst = np.zeros((rows, cols), dtype=np.float64, order="C")
    for i in range(rows):
        for j in range(cols):
            dst[i, j] = src[i * interface.cols + j]
    return dst