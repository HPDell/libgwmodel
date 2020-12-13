cimport cbase

cdef class Distance:
    cdef CGwmDistance* _c_instance
    
    def __cinit__(self, bint geographical):
        self._c_instance = cbase.gwmodel_create_crs_distance(geographical)
    
    def __dealloc__(self):
        cbase.gwmodel_delete_crs_distance(self)

    @property
    def geographical(self):
        return self._c_instance.geographical


cdef class Weight:
    cdef CGwmWeight* _c_instance
    
    def __cinit__(self, double size, bint adaptive, cbase.KernelFunctionType kernel):
        self._c_instance = cbase.gwmodel_create_bandwidth_weight(size, adaptive, kernel)
    
    def __dealloc__(self):
        cbase.gwmodel_delete_bandwidth_weight(self)

    @property
    def size(self):
        return self._c_instance.size

    @property
    def adaptive(self):
        return self._c_instance.adaptive

    @property
    def kernel(self):
        return self._c_instance.kernel


cdef class SpatialWeight:
    cdef CGwmSpatialWeight* _c_instance
    
    def __cinit__(self, Distance distance, Weight weight):
        self.distance = distance
        self.weight = weight
        self._c_instance = cbase.gwmodel_create_spatial_weight(distance._c_instance, weight._c_instance)
    
    def __dealloc__(self):
        cbase.gwmodel_delete_spatial_weight(self)
