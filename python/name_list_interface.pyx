from name_list_interface cimport GwmNameInterface, GwmNameListInterface, gwmodel_delete_string_list
from libc.stdlib cimport malloc
from libc.string cimport strncpy

cdef class NameListInterface:    
    def __cinit__(self, list names):
        cdef size_t size = len(names)
        cdef GwmNameInterface* data_ptr = <GwmNameInterface*>malloc(size * sizeof(GwmNameInterface))
        cdef char[:] src
        for i in range(size):
            src = names[i]
            strncpy(data_ptr[i], &src[0], len(src))
        self._c_instance = GwmNameListInterface(size, data_ptr)
    
    def __dealloc__(self):
        gwmodel_delete_string_list(&self._c_instance)
    
    @property
    def size(self):
        return self._c_instance.size