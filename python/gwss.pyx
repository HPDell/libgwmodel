from gwss cimport *
from simple_layer cimport SimpleLayer, CGwmSimpleLayer
from spatial_weight cimport Distance, Weight, SpatialWeight
from mat_interface cimport MatInterface, mat2numpy, mat2interface
from variable_interface cimport VariableListInterface
from name_list_interface cimport NameListInterface
from cython.view cimport array as cvarray

cdef class GWSS:
    def __cinit__(self, SimpleLayer source_layer, SpatialWeight spatial_weight, VariableListInterface variables, bint quantile, bint first_only):
        self._c_instance = gwmodel_create_gwss_algorithm()
        gwmodel_set_gwss_source_layer(self._c_instance, source_layer._c_instance)
        gwmodel_set_gwss_spatial_weight(self._c_instance, spatial_weight._c_instance)
        gwmodel_set_gwss_variables(self._c_instance, variables._c_instance)
        gwmodel_set_gwss_options(self._c_instance, quantile, first_only)
    
    def __dealloc__(self):
        gwmodel_delete_gwss_algorithm(self._c_instance)
    
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = gwmodel_get_gwss_result_layer(self._c_instance)
        cdef MatInterface points = mat2interface(gwmodel_get_simple_layer_points(layer))
        cdef MatInterface data = mat2interface(gwmodel_get_simple_layer_data(layer))
        return SimpleLayer(points, data, NameListInterface([]))
    
    def local_mean(self):
        return mat2numpy(gwmodel_get_gwss_local_mean(self._c_instance))

    def local_sdev(self):
        return mat2numpy(gwmodel_get_gwss_local_sdev(self._c_instance))

    def local_var(self):
        return mat2numpy(gwmodel_get_gwss_local_var(self._c_instance))

    def local_skew(self):
        return mat2numpy(gwmodel_get_gwss_local_skew(self._c_instance))

    def local_cv(self):
        return mat2numpy(gwmodel_get_gwss_local_cv(self._c_instance))

    def local_cov(self):
        return mat2numpy(gwmodel_get_gwss_local_cov(self._c_instance))

    def local_corr(self):
        return mat2numpy(gwmodel_get_gwss_local_corr(self._c_instance))

    def local_spearman_rho(self):
        return mat2numpy(gwmodel_get_gwss_local_spearman_rho(self._c_instance))

    def local_median(self):
        return mat2numpy(gwmodel_get_gwss_local_median(self._c_instance))

    def local_iqr(self):
        return mat2numpy(gwmodel_get_gwss_local_iqr(self._c_instance))

    def local_qi(self):
        return mat2numpy(gwmodel_get_gwss_local_qi(self._c_instance))
    
    def enable_openmp(self, int threads):
        gwmodel_set_gwss_openmp(self._c_instance, threads)
    
    def run(self):
        gwmodel_run_gwss(self._c_instance)
