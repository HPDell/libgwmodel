__all__ = ["NameListInterface"]

from cbase cimport GwmNameInterface, GwmNameListInterface, gwmodel_delete_string_list

cdef class NameListInterface:
    cdef GwmNameListInterface _c_instance