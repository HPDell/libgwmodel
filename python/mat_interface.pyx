from cbase cimport GwmMatInterface
from libc.stdlib cimport malloc, free
from numpy import ndarray

cdef class MatInterface:
    cdef GwmMatInterface _c_instance

    def __cinit__(self, rows, cols, double[:, ::1] array):
        self._c_instance = GwmMatInterface(rows, cols, &array[0, 0])


def numpy2mat(double[:, ::1] array):
    cdef unsigned long long rows = array.shape[0]
    cdef unsigned long long cols = array.shape[1]
    return MatInterface(rows, cols, array)


cdef mat2numpy(GwmMatInterface interface):
    cdef double[:, ::1] src = interface.data, dst = np.zeros((interface.rows, interface.cols), dtype=double, order="C")
    dst[:, :] = src
    return dst