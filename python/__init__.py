import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
from typing import List
from .mat_interface import numpy2mat as cyg_numpy2mat
from .name_list_interface import NameListInterface as cygNameListInterface
from .simple_layer import SimpleLayer as cygSimpleLayer
from . import spatial_weight as cyg_sw
from . import variable_interface as cyg_varlist
from .gwss import GWSS as cygGWSS
from . import regression_diagnostic as cygDiagnostic
from .gwr_basic import GWRBasic as cygGWRBasic

KERNEL_GAUSSIAN = 0

CRITERION_AIC = 0
CRITERION_CV = 1


def sdf_to_layer(sdf: gp.GeoDataFrame, variables: List[str]):
    """
    Convert GeoDataFrame to SimpleLayer
    """
    cyg_coord = cyg_numpy2mat(np.asfortranarray([[x, y] for x, y in zip(sdf.centroid.x, sdf.centroid.y)]))
    cyg_data = cyg_numpy2mat(np.asfortranarray(sdf[variables]))
    cyg_fields = cygNameListInterface([x.encode() for x in sdf.columns])
    return cygSimpleLayer(cyg_coord, cyg_data, cyg_fields)


def layer_to_sdf(layer: cygSimpleLayer, geometry: gp.GeoSeries=None):
    """
    Convert GeoDataFrame to SimpleLayer
    """
    result_geom = layer.points  # type: np.ndarray
    result_attr = layer.data
    result_field = [f.decode() for f in layer.fields]
    result_data = {f: result_attr[:, i] for i, f in enumerate(result_field)}
    return gp.GeoDataFrame({
        **result_data,
        "geometry": gp.GeoSeries([Point(result_geom[p, 0], result_geom[p, 1]) for p in range(result_geom.shape[0])])
    }) if geometry is None else gp.GeoDataFrame({
        **result_data,
        "geometry": geometry
    })
    

class GWSS:
    """
    GWSS python high api class.
    """

    def __init__(self, sdf: gp.GeoDataFrame, variables: List[str], bw, adaptive=True, kernel=KERNEL_GAUSSIAN, longlat=True, quantile=False, first_only=False):
        """
        docstring
        """
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.sdf = sdf
        self.variables = variables
        self.bw = bw
        self.kernel = kernel
        self.adaptive = adaptive
        self.longlat = longlat
        self.result_layer = None
        self.quantile = quantile
        self.first_only = first_only

    def fit(self):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        cyg_data_layer = sdf_to_layer(self.sdf, self.variables)
        cyg_distance = cyg_sw.Distance(self.longlat)
        cyg_weight = cyg_sw.Weight(self.bw, self.adaptive, self.kernel)
        cyg_spatial_weight = cyg_sw.SpatialWeight(cyg_distance, cyg_weight)
        cyg_in_vars = cyg_varlist.VariableListInterface([cyg_varlist.VariableInterface(i, True, n.encode()) for i, n in enumerate(self.variables)])
        ''' Create cython GWSS
        '''
        cyg_gwss = cygGWSS(cyg_data_layer, cyg_spatial_weight, cyg_in_vars, self.quantile, self.first_only)
        cyg_gwss.run()
        self.result_layer = layer_to_sdf(cyg_gwss.result_layer, self.sdf.geometry)
        return self


class GWRBasic:
    """
    Basic GWR python high api class.
    """

    def __init__(self, sdf: gp.GeoDataFrame, depen_var: str, indep_vars: List[str], bw=None, adaptive=True, kernel=KERNEL_GAUSSIAN, longlat=True):
        """
        docstring
        """
        if not isinstance(sdf, gp.GeoDataFrame):
            raise ValueError("sdf must be a GeoDataFrame")
        self.sdf = sdf
        self.depen_var = depen_var
        self.indep_vars = indep_vars
        self.bw = bw
        self.kernel = kernel
        self.adaptive = adaptive
        self.longlat = longlat
        self.result_layer = None
        self.diagnostic = None
        self.bandwidth_select_criterions = None
        self.indep_var_select_criterions = None
    
    def fit(self, hatmatrix=True, optimize_bw=None, optimize_var=None, multithreads=None):
        """
        Run algorithm and return result
        """
        ''' Extract data
        '''
        cyg_data_layer = sdf_to_layer(self.sdf, [self.depen_var] + self.indep_vars)
        cyg_distance = cyg_sw.Distance(self.longlat)
        cyg_weight = cyg_sw.Weight((self.bw if self.bw is not None else 0.0), self.adaptive, self.kernel)
        cyg_spatial_weight = cyg_sw.SpatialWeight(cyg_distance, cyg_weight)
        cyg_depen_var = cyg_varlist.VariableInterface(0, True, self.depen_var.encode())
        cyg_indep_vars = cyg_varlist.VariableListInterface([cyg_varlist.VariableInterface(i + 1, True, n.encode()) for i, n in enumerate(self.indep_vars)])
        ''' Create cython GWR
        '''
        cyg_gwr_basic = cygGWRBasic(cyg_data_layer, cyg_spatial_weight, cyg_indep_vars, cyg_depen_var, hatmatrix)
        if self.bw is None:
            cyg_gwr_basic.set_bandwidth_autoselection((optimize_bw if optimize_bw is not None else CRITERION_CV))
        elif optimize_bw is not None:
            if optimize_bw == CRITERION_AIC or optimize_bw == CRITERION_CV:
                cyg_gwr_basic.set_bandwidth_autoselection(optimize_bw)
            else:
                raise ValueError("optimize_bw must be CRITERION_AIC(0) or CRITERION_CV(1)")
        if optimize_var is not None:
            if isinstance(optimize_var, float) and optimize_var > 0:
                cyg_gwr_basic.set_indep_vars_autoselection(optimize_var)
            else:
                raise ValueError("optimize_var must be a positive real number")
        if multithreads is not None:
            if isinstance(multithreads, int) and multithreads > 0:
                cyg_gwr_basic.enable_openmp(multithreads)
            else:
                raise ValueError("multithreads must be a positive integer")
        cyg_gwr_basic.run()
        if self.bw is None or optimize_bw is not None:
            self.bw = cyg_gwr_basic.bandwidth
            self.bandwidth_select_criterions = cyg_gwr_basic.bandwidth_select_criterions
        if optimize_var is not None:
            self.indep_vars = [v.decode() for v in cyg_gwr_basic.indep_vars]
            self.indep_var_select_criterions = cyg_gwr_basic.indep_var_select_criterions
        ''' Get result layer
        '''
        self.result_layer = layer_to_sdf(cyg_gwr_basic.result_layer, self.sdf.geometry)
        if hatmatrix:
            self.diagnostic = cyg_gwr_basic.diagnostic
        return self

    def predict(self, targets: gp.GeoDataFrame, multithreads=None):
        """
        Predict
        """
        if self.bw is None:
            raise ValueError("Bandwidth cannot be None when predicting")
        ''' Extract data
        '''
        cyg_source_layer = sdf_to_layer(self.sdf, [self.depen_var] + self.indep_vars)
        cyg_predict_layer = sdf_to_layer(targets, self.indep_vars)
        cyg_distance = cyg_sw.Distance(self.longlat)
        cyg_weight = cyg_sw.Weight(self.bw, self.adaptive, self.kernel)
        cyg_spatial_weight = cyg_sw.SpatialWeight(cyg_distance, cyg_weight)
        cyg_depen_var = cyg_varlist.VariableInterface(0, True, self.depen_var.encode())
        cyg_indep_vars = cyg_varlist.VariableListInterface([cyg_varlist.VariableInterface(i + 1, True, n.encode()) for i, n in enumerate(self.indep_vars)])
        ''' Create cython GWR
        '''
        cyg_gwr_basic = cygGWRBasic(cyg_source_layer, cyg_spatial_weight, cyg_indep_vars, cyg_depen_var, False)
        cyg_gwr_basic.set_predict_layer(cyg_predict_layer)
        if multithreads is not None:
            if isinstance(multithreads, int) and multithreads > 0:
                cyg_gwr_basic.enable_openmp(multithreads)
            else:
                raise ValueError("multithreads must be a positive integer")
        ''' Get result layer
        '''
        cyg_gwr_basic.run()
        return layer_to_sdf(cyg_gwr_basic.result_layer, targets.geometry)
