cimport cbase
from libc.stdlib cimport malloc, free
from numpy import ndarray

cdef class VariableInterface:
    cdef GwmVariableInterface* _c_instance
    
    def __cinit__(self, int index, bint numeric, char[:] name):
        self._c_instance = <cbase.GwmVariableInterface*>malloc(sizeof(cbase.GwmVariableInterface*))
        self._c_instance.index = index
        self._c_instance.numeric = numeric
        char[:] name_dst = self._c_instance.name
        name_dst[...] = name

    def __dealloc__(self)
        free(self._c_instance)


cdef class VariableListInterface:
    cdef GwmVariableListInterface* _c_instance
    
    def __cinit__(self, int size, list variables):
        self._c_instance = <cbase.GwmVariableListInterface*>malloc(sizeof(cbase.GwmVariableListInterface*))
        self._c_instance.size = size
        cbase.GwmVariableInterface* items = <cbase.GwmVariableInterface*>malloc(size * sizeof(cbase.GwmVariableInterface))
        for i in range(size):
            items[i] = variables[i]._c_instance
        self._c_instance.items = items

    def __dealloc__(self)
        free(self._c_instance)
    
    @property
    def size(self)
        return self._c_instance.size