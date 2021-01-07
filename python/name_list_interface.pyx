from name_list_interface cimport GwmNameInterface, GwmNameListInterface, gwmodel_delete_string_list
from libc.stdlib cimport malloc
from libc.string cimport strcpy, strlen
from libcpp.string cimport string

cdef class NameListInterface:    
    def __cinit__(self, list names):
        cdef size_t size = len(names)
        cdef GwmNameInterface* data_ptr = <GwmNameInterface*>malloc(size * sizeof(GwmNameInterface))
        cdef const unsigned char[:] src
        cdef int i
        for i in range(size):
            src = names[i]
            strcpy(data_ptr[i], <const char*>(&src[0]))
        self._c_instance = GwmNameListInterface(size, data_ptr)
    
    def __dealloc__(self):
        gwmodel_delete_string_list(&self._c_instance)
    
    @property
    def size(self):
        return self._c_instance.size

cdef names2list(GwmNameListInterface instance):
    name_list = []
    cdef int size = instance.size, i
    cdef string name
    for i in range(size):
        name = string(&instance.items[i][0])
        py_str = name
        name_list.append(py_str)
    return name_list
