from cbase cimport CGwmSimpleLayer
from cbase cimport gwmodel_create_simple_layer, gwmodel_delete_simple_layer, gwmodel_get_simple_layer_points, gwmodel_get_simple_layer_data, gwmodel_get_simple_layer_fields

cdef class SimpleLayer:
    cdef CGwmSimpleLayer* _c_instance