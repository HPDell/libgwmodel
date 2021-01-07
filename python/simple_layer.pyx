from mat_interface cimport MatInterface, numpy2mat, mat2numpy
from name_list_interface cimport NameListInterface, names2list
from simple_layer cimport gwmodel_create_simple_layer, gwmodel_delete_simple_layer, gwmodel_get_simple_layer_points, gwmodel_get_simple_layer_data, gwmodel_get_simple_layer_fields

cdef class SimpleLayer:
    def __cinit__(self, MatInterface points, MatInterface data, NameListInterface fields):
        self._c_instance = gwmodel_create_simple_layer(points._c_instance, data._c_instance, fields._c_instance)
    
    def __dealloc__(self):
        gwmodel_delete_simple_layer(self._c_instance)
    
    @property
    def points(self):
        return mat2numpy(gwmodel_get_simple_layer_points(self._c_instance))
    
    # @points.setter
    # def points(self, MatInterface value):
    #     self._c_instance.points = value._c_instance

    @property
    def data(self):
        return mat2numpy(gwmodel_get_simple_layer_data(self._c_instance))
    
    # @data.setter
    # def data(self, MatInterface value):
    #     self._c_instance.data = value._c_instance

    @property
    def fields(self):
        return names2list(gwmodel_get_simple_layer_fields(self._c_instance))
    
    # @fields.setter
    # def fields(self, NameListInterface value):
    #     self._c_instance.fields = value._c_instance