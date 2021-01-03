from variable_interface cimport GwmVariableInterface, GwmVariableListInterface, gwmodel_delete_variable_list
from libc.stdlib cimport malloc
from libc.string cimport strncpy, strcpy, memcpy

cdef class VariableInterface:    
    def __cinit__(self, int index, bint numeric, char[:] name):
        self._c_instance = GwmVariableInterface()
        self._c_instance.index = index
        self._c_instance.numeric = numeric
        strncpy(self._c_instance.name, &name[0], len(name))

    def __dealloc__(self):
        strcpy(self._c_instance.name, "")


cdef class VariableListInterface:
    def __cinit__(self, int size, VariableInterface[:] variables):
        cdef GwmVariableInterface* items = <GwmVariableInterface*>malloc(size * sizeof(GwmVariableInterface))
        cdef int i
        for i in range(size):
            items[i] = variables[i]._c_instance
        self._c_instance = GwmVariableListInterface(size, items)

    def __dealloc__(self):
        gwmodel_delete_variable_list(&self._c_instance)
    