cimport cbase
from cbase cimport MatInterface, numpy2mat, mat2numpy
from cbase cimport NameListInterface

cdef class SimpleLayer:
    cdef cbase.CGwmSimpleLayer* _c_instance

    def __cinit__(self, MatInterface points, MatInterface data, NameListInterface fields):
        self._c_instance = cbase.gwmodel_create_simple_layer(points._c_instance, data._c_instance, fields._c_instance)
    
    def __dealloc__(self):
        cbase.gwmodel_delete_simple_layer(self._c_instance)
    
    @property
    def points(self):
        return mat2numpy(cbase.gwmodel_get_simple_layer_points(self._c_instance))
    
    # @points.setter
    # def points(self, MatInterface value):
    #     self._c_instance.points = value._c_instance

    @property
    def data(self):
        return mat2numpy(cbase.gwmodel_get_simple_layer_data(self._c_instance))
    
    # @data.setter
    # def data(self, MatInterface value):
    #     self._c_instance.data = value._c_instance

    # @property
    # def fields(self):
    #     return mat2numpy(cbase.gwmodel_get_simple_layer_fields(self._c_instance))
    
    # @fields.setter
    # def fields(self, NameListInterface value):
    #     self._c_instance.fields = value._c_instance