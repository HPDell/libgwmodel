from cbase cimport CGwmGWRBasic, BandwidthSelectionCriterionType, GwmBandwidthKernelInterface, CGwmSpatialWeight, GwmRegressionDiagnostic, GwmMatInterface
from cbase cimport GwmVariablesCriterionPairInterface, GwmVariablesCriterionListInterface
from cbase cimport GwmBandwidthCriterionPairInterface, GwmBandwidthCriterionListInterface
from cbase cimport gwmodel_get_simple_layer_points
from cbase cimport gwmodel_get_simple_layer_data
from cbase cimport gwmodel_get_simple_layer_fields
from cbase cimport gwmodel_create_gwr_algorithm
from cbase cimport gwmodel_delete_gwr_algorithm
from cbase cimport gwmodel_delete_spatial_weight
from cbase cimport gwmodel_set_gwr_source_layer
from cbase cimport gwmodel_set_gwr_spatial_weight
from cbase cimport gwmodel_set_gwr_dependent_variable
from cbase cimport gwmodel_set_gwr_independent_variable
from cbase cimport gwmodel_set_gwr_predict_layer
from cbase cimport gwmodel_set_gwr_bandwidth_autoselection
from cbase cimport gwmodel_set_gwr_indep_vars_autoselection
from cbase cimport gwmodel_set_gwr_options
from cbase cimport gwmodel_set_gwr_openmp
from cbase cimport gwmodel_get_gwr_spatial_weight
from cbase cimport gwmodel_get_gwr_result_layer
from cbase cimport gwmodel_get_gwr_coefficients
from cbase cimport gwmodel_get_gwr_diagnostic
from cbase cimport gwmodel_get_gwr_bandwidth_criterions
from cbase cimport gwmodel_get_gwr_indep_var_criterions
from cbase cimport gwmodel_run_gwr
from cbase cimport gwmodel_get_spatial_bandwidth_weight

cdef class GWRBasic:
    cdef CGwmGWRBasic* _c_instance