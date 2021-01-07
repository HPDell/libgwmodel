from spatial_weight cimport KernelFunctionType
from spatial_weight cimport gwmodel_create_crs_distance, gwmodel_delete_crs_distance
from spatial_weight cimport gwmodel_create_bandwidth_weight, gwmodel_delete_bandwidth_weight
from spatial_weight cimport gwmodel_create_spatial_weight, gwmodel_delete_spatial_weight

cdef class Distance:    
    def __cinit__(self, bint geographical):
        self._c_instance = gwmodel_create_crs_distance(geographical)
    
    def __dealloc__(self):
        gwmodel_delete_crs_distance(self._c_instance)


cdef class Weight:    
    def __cinit__(self, double size, bint adaptive, KernelFunctionType kernel):
        self._c_instance = gwmodel_create_bandwidth_weight(size, adaptive, kernel)
    
    def __dealloc__(self):
        gwmodel_delete_bandwidth_weight(self._c_instance)


cdef class SpatialWeight:    
    def __cinit__(self, Distance distance, Weight weight):
        self._c_instance = gwmodel_create_spatial_weight(distance._c_instance, weight._c_instance)
    
    def __dealloc__(self):
        gwmodel_delete_spatial_weight(self._c_instance)
