from gwr_basic cimport *
from simple_layer cimport SimpleLayer, CGwmSimpleLayer
from spatial_weight cimport Distance, Weight, SpatialWeight
from mat_interface cimport MatInterface, mat2numpy, mat2interface
from variable_interface cimport VariableInterface, VariableListInterface, GwmVariableListInterface, GwmVariableInterface
from name_list_interface cimport NameListInterface, names2list
from regression_diagnostic cimport RegressionDiagnostic
from cython.view cimport array as cvarray
from libcpp.string cimport string

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
    
    @property
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = gwmodel_get_gwr_result_layer(self._c_instance)
        cdef MatInterface points = mat2interface(gwmodel_get_simple_layer_points(layer))
        cdef MatInterface data = mat2interface(gwmodel_get_simple_layer_data(layer))
        return SimpleLayer(points, data, NameListInterface(names2list(gwmodel_get_simple_layer_fields(layer))))
    
    @property
    def coefficients(self):
        return mat2numpy(gwmodel_get_gwr_coefficients(self._c_instance))

    @property
    def diagnostic(self):
        return RegressionDiagnostic.wrap(gwmodel_get_gwr_diagnostic(self._c_instance))
    
    @property
    def bandwidth_select_criterions(self):
        criterion_list = []
        cdef GwmBandwidthCriterionListInterface interf = gwmodel_get_gwr_bandwidth_criterions(self._c_instance)
        cdef int size = interf.size, p
        for p in range(size):
            bw = interf.items[p].bandwidth
            value = interf.items[p].criterion
            criterion_list.append((bw, value))
        return criterion_list
    
    @property
    def indep_var_select_criterions(self):
        criterion_list = []
        cdef GwmVariablesCriterionListInterface interf = gwmodel_get_gwr_indep_var_criterions(self._c_instance)
        cdef int size = interf.size, p, v
        cdef double value = 0.0
        cdef GwmVariableListInterface var_list
        cdef GwmVariableInterface variable
        cdef string name
        for p in range(size):
            var_list = interf.items[p].variables
            value = interf.items[p].criterion
            var_names = []
            for v in range(var_list.size):
                name = string(&(var_list.items[v].name[0]))
                py_str = name.decode()
                var_names.append(py_str)
            criterion_list.append((var_names, value))
        return criterion_list

    def set_predict_layer(self, SimpleLayer predict_layer):
        gwmodel_set_gwr_predict_layer(self._c_instance, predict_layer._c_instance)

    def set_bandwidth_autoselection(self, BandwidthSelectionCriterionType criterion):
        gwmodel_set_gwr_bandwidth_autoselection(self._c_instance, criterion)

    def set_indep_vars_autoselection(self, double threshold):
        gwmodel_set_gwr_indep_vars_autoselection(self._c_instance, threshold)
    
    def enable_openmp(self, int threads):
        gwmodel_set_gwr_openmp(self._c_instance, threads)
        
    def run(self):
        gwmodel_run_gwr(self._c_instance)