cimport cbase
from cbase cimport GwmNameInterface
from libc.stdlib cimport malloc
from libc.string cimport strncpy

cdef class NameListInterface:
    cdef cbase.GwmNameListInterface _c_instance
    
    def __cinit__(self, list names):
        cdef size_t size = len(names)
        cdef GwmNameInterface* data_ptr = <GwmNameInterface*>malloc(size * sizeof(GwmNameInterface))
        cdef char[:] src
        for i in range(size):
            src = names[i]
            strncpy(data_ptr[i], &src[0], len(src))
        self._c_instance = cbase.GwmNameListInterface(size, data_ptr)
    
    def __dealloc__(self):
        cbase.gwmodel_delete_string_list(&self._c_instance)
    
    @property
    def size(self):
        return self._c_instance.size