cimport cbase
from libc.stdlib cimport malloc, free
from numpy import ndarray

cdef class NameListInterface:
    cdef GwmNameListInterface* _c_instance
    
    def __cinit__(self, list names):
        self._c_instance = <cbase.GwmNameListInterface*>malloc(sizeof(cbase.GwmNameListInterface*))
        size = len(names)
        self._c_instance.size = size
        cdef cbase.GwmNameInterface* data_ptr = <cbase.GwmNameInterface*>malloc(size * sizeof(GwmNameInterface))
        for i in range(size):
            cdef char[:] dst = data_ptr[i]
            cdef char[:] src = names[i]
            dst[...] = src
        self._c_instance.data = data_ptr
    
    def __dealloc__(self)
        free(self._c_instance)
    
    @property
    def size(self)
        return self._c_instance.size