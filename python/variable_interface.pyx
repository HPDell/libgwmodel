from variable_interface cimport GwmVariableInterface, GwmVariableListInterface, gwmodel_delete_variable_list
from libc.stdlib cimport malloc
from libc.string cimport strncpy, strcpy, strlen

cdef class VariableInterface:
    def __cinit__(self, int index, bint numeric, const unsigned char[:] name):
        self._c_instance = GwmVariableInterface()
        self._c_instance.index = index
        self._c_instance.isNumeric = numeric
        strncpy(self._c_instance.name, <const char*>(&name[0]), len(name))

    def __dealloc__(self):
        strcpy(self._c_instance.name, "")


cdef class VariableListInterface:
    def __cinit__(self, list variables):
        cdef int size = len(variables), i
        cdef GwmVariableInterface* items = <GwmVariableInterface*>malloc(size * sizeof(GwmVariableInterface))
        cdef VariableInterface var
        for i in range(size):
            var = (<VariableInterface>(variables[i]))
            items[i].index = var._c_instance.index
            items[i].isNumeric = var._c_instance.isNumeric
            strncpy(items[i].name, var._c_instance.name, strlen(var._c_instance.name))
        self._c_instance = GwmVariableListInterface(size, items)

    def __dealloc__(self):
        gwmodel_delete_variable_list(&self._c_instance)
    