from cbase cimport CGwmDistance, CGwmWeight, CGwmSpatialWeight, KernelFunctionType, GwmBandwidthKernelInterface
from cbase cimport gwmodel_create_crs_distance, gwmodel_delete_crs_distance
from cbase cimport gwmodel_create_bandwidth_weight, gwmodel_delete_bandwidth_weight
from cbase cimport gwmodel_create_spatial_weight, gwmodel_delete_spatial_weight

cdef class Distance:
    cdef CGwmDistance* _c_instance

cdef class Weight:
    cdef CGwmWeight* _c_instance

cdef class SpatialWeight:
    cdef CGwmSpatialWeight* _c_instance
