from gwr_basic cimport *
from simple_layer cimport SimpleLayer, CGwmSimpleLayer
from spatial_weight cimport Distance, Weight, SpatialWeight
from mat_interface cimport MatInterface, mat2numpy, mat2interface
from variable_interface cimport VariableInterface, VariableListInterface
from name_list_interface cimport NameListInterface, names2list
from cython.view cimport array as cvarray

cdef class GWRBasic:
    def __cinit__(self, SimpleLayer source_layer, SpatialWeight spatial_weight, VariableListInterface indep_variables, VariableInterface dep_variable, bint hat_matrix):
        self._c_instance = gwmodel_create_gwr_algorithm()
        gwmodel_set_gwr_source_layer(self._c_instance, source_layer._c_instance)
        gwmodel_set_gwr_spatial_weight(self._c_instance, spatial_weight._c_instance)
        gwmodel_set_gwr_independent_variable(self._c_instance, indep_variables._c_instance)
        gwmodel_set_gwr_dependent_variable(self._c_instance, dep_variable._c_instance)
        gwmodel_set_gwr_options(self._c_instance, hat_matrix)
    
    def __dealloc__(self):
        gwmodel_delete_gwr_algorithm(self._c_instance) 
    
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = gwmodel_get_gwr_result_layer(self._c_instance)
        cdef MatInterface points = mat2interface(gwmodel_get_simple_layer_points(layer))
        cdef MatInterface data = mat2interface(gwmodel_get_simple_layer_data(layer))
        return SimpleLayer(points, data, NameListInterface(names2list(gwmodel_get_simple_layer_fields(layer))))
    
    def coefficients(self):
        return mat2numpy(gwmodel_get_gwr_coefficients(self._c_instance))
        
    def run(self):
        gwmodel_run_gwr(self._c_instance)