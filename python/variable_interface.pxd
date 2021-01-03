from cbase cimport GwmVariableInterface, GwmVariableListInterface
from cbase cimport gwmodel_delete_variable_list

cdef class VariableInterface:
    cdef GwmVariableInterface _c_instance

cdef class VariableListInterface:
    cdef GwmVariableListInterface _c_instance