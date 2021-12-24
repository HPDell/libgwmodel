# distutils: language = c++

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
import numpy as np
cimport numpy as np
import geopandas as gp
from .gwmodel cimport mat, cube
from .gwmodel cimport CGwmSimpleLayer
from .gwmodel cimport GwmVariable
from .gwmodel cimport CGwmDistance, CGwmCRSDistance
from .gwmodel cimport CGwmWeight, CGwmBandwidthWeight
from .gwmodel cimport CGwmSpatialWeight
from .gwmodel cimport CGwmGWRBasic, CGwmGWRBase, CGwmGWPCA, CGwmGWSS, GwmRegressionDiagnostic
from .gwmodel cimport BandwidthCriterionList, VariablesCriterionList
from .gwmodel cimport ParallelType


cdef mat numpy2mat(double[::1, :] array):
    cdef unsigned long long rows = array.shape[0]
    cdef unsigned long long cols = array.shape[1]
    return mat(&array[0, 0], rows, cols)


cdef mat2numpy(const mat& arma_mat):
    cdef const double* src = arma_mat.memptr()
    cdef unsigned long long rows = arma_mat.n_rows
    cdef unsigned long long cols = arma_mat.n_cols
    result = np.zeros((rows, cols), dtype=np.float64, order="F")
    cdef double[::1, :] dst = result
    cdef unsigned long long i
    for i in range(cols):
        for j in range(rows):
            dst[j, i] = src[i * rows + j]
    return result

cdef cube2numpy(const cube& arma_cube):
    cdef const double* src = arma_cube.memptr()
    cdef unsigned long long rows = arma_cube.n_rows
    cdef unsigned long long cols = arma_cube.n_cols
    cdef unsigned long long slices = arma_cube.n_slices
    result = np.zeros((slices, rows, cols), dtype=np.float64, order="C")
    cdef double[:, :, ::1] dst = result
    cdef unsigned long long i, j, k
    for k in range(slices):
        for j in range(cols):
            for i in range(rows):
                dst[k, i, j] = src[k * cols * rows + j * rows + i]
    return result

cdef vector[string] name_list2vector(list names):
    cdef size_t size = len(names)
    cdef int i
    cdef string cname
    cdef vector[string] cnames
    for i in range(size):
        cname = names[i]
        cnames.push_back(cname)
    return cnames


cdef name_vector2list(vector[string] instance):
    names = []
    cdef int size = instance.size(), i
    cdef string name
    for i in range(size):
        names.append(instance.at(i))
    return names


cdef class CySimpleLayer:
    cdef CGwmSimpleLayer _c_instance

    def __cinit__(self, double[::1, :] points, double[::1, :] data, list fields):
        self._c_instance = CGwmSimpleLayer(numpy2mat(points), numpy2mat(data), name_list2vector(fields))
    
    @property
    def points(self):
        return mat2numpy(self._c_instance.points())
    
    # @points.setter
    # def points(self, double[::1, :] value):
    #     self._c_instance.points = numpy2mat(value)

    @property
    def data(self):
        return mat2numpy(self._c_instance.data())
    
    # @data.setter
    # def data(self, double[::1, :] value):
    #     self._c_instance.data = numpy2mat(value)

    @property
    def fields(self):
        return name_vector2list(self._c_instance.fields())
    
    # @fields.setter
    # def fields(self, list value):
    #     self._c_instance.fields = name_list2vector(value)
    
    @property
    def featureCount(self):
        return self._c_instance.featureCount()


cdef class CyVariable:
    cdef GwmVariable _c_instance

    def __cinit__(self, int i, bint numeric, const unsigned char[:] n):
        self._c_instance = GwmVariable(i, numeric, bytes(n))
    
    @property
    def index(self):
        return self._c_instance.index
    
    # @index.setter
    # def index(self, int value):
    #     self._c_instance.index = value
    
    @property
    def is_numeric(self):
        return self._c_instance.isNumeric

    # @is_numeric.setter
    # def is_numeric(self, bint value):
    #     self._c_instance.isNumeric = value
    
    @property
    def name(self):
        return self._c_instance.name

    # @name.setter
    # def name(self, str value):
    #     self._c_instance.name = value


cdef class CyVariableList:
    cdef vector[GwmVariable] _c_instance

    def __cinit__(self, list var_list):
        cdef size_t size = len(var_list)
        cdef int i
        cdef GwmVariable cvar
        cdef vector[GwmVariable] cvars
        for i in range(size):
            cvar = (<CyVariable>(var_list[i]))._c_instance
            cvars.push_back(cvar)
        self._c_instance = cvars


cdef class CyDistance:
    __type = None
    cdef CGwmDistance* _c_instance

    def distance_type(self):
        return self.__type


cdef class CyCRSDistance(CyDistance):
    __type = "CRS"

    def __cinit__(self, bint geographic):
        self._c_instance = new CGwmCRSDistance(geographic);


cdef class CyWeight:
    __type = None
    cdef CGwmWeight* _c_instance

    def weight_type(self):
        return self.__type


cdef class CyBandwidthWeight(CyWeight):
    __type = "Bandwidth"
    
    def __cinit__(self, double size, bint adaptive, int kernel):
        self._c_instance = new CGwmBandwidthWeight(size, adaptive, <CGwmBandwidthWeight.KernelFunctionType>kernel)


cdef class CyGWRBasic:
    cdef CGwmGWRBasic* _c_instance

    def __cinit__(self, CySimpleLayer layer, CyVariable depen_var, CyVariableList indep_var_list, CyWeight weight, CyDistance distance, bint hatmatrix):
        self._c_instance = new CGwmGWRBasic()
        self._c_instance.setSourceLayer(layer._c_instance)
        self._c_instance.setDependentVariable(depen_var._c_instance)
        self._c_instance.setIndependentVariables(indep_var_list._c_instance)
        cdef CGwmSpatialWeight spatial = CGwmSpatialWeight(weight._c_instance, distance._c_instance)
        self._c_instance.setSpatialWeight(spatial)
        self._c_instance.setHasHatMatrix(hatmatrix)
    
    def enable_bandwidth_autoselection(self, int criterion):
        self._c_instance.setIsAutoselectBandwidth(True)
        self._c_instance.setBandwidthSelectionCriterion(<CGwmGWRBasic.BandwidthSelectionCriterionType>criterion)
    
    def enable_indep_var_autoselection(self, double threshold):
        self._c_instance.setIsAutoselectIndepVars(True)
        self._c_instance.setIndepVarSelectionThreshold(threshold)
    
    def enable_openmp(self, int threads):
        self._c_instance.setParallelType(ParallelType.OpenMP)
        self._c_instance.setOmpThreadNum(threads)
    
    def set_predict_layer(self, CySimpleLayer predict_layer):
        self._c_instance.setPredictLayer(&predict_layer._c_instance)

    def run(self):
        self._c_instance.run()
    
    def diagnostic(self):
        cdef GwmRegressionDiagnostic diag = self._c_instance.diagnostic()
        return {
            "RSS": diag.RSS,
            "AIC": diag.AIC,
            "AICc": diag.AICc,
            "ENP": diag.ENP,
            "EDF": diag.EDF,
            "RSquare": diag.RSquare,
            "RSquareAdjust": diag.RSquareAdjust
        }
    
    def bandwidth_select_criterions(self):
        cdef BandwidthCriterionList criterion_c = self._c_instance.bandwidthSelectionCriterionList()
        criterion_py = []
        cdef unsigned long long size = criterion_c.size()
        cdef pair[double, double] item
        for i in range(size):
            item = criterion_c.at(i)
            criterion_py.append((item.first, item.second))
        return criterion_py

    def indep_var_select_criterions(self):
        cdef VariablesCriterionList criterion_c = self._c_instance.indepVarsSelectionCriterionList()
        criterion_py = []
        cdef unsigned long long criterion_size = criterion_c.size()
        cdef pair[vector[GwmVariable],double] item
        cdef vector[GwmVariable] var_list
        cdef unsigned long long var_size
        cdef string var
        for i in range(criterion_size):
            item = criterion_c.at(i)
            var_list = item.first
            var_size = var_list.size()
            var_list_py = []
            for j in range(var_size):
                var = var_list.at(j).name
                var_list_py.append(var.decode())
            criterion_py.append((var_list_py, item.second))
        return criterion_py
    
    def bandwidth(self):
        cdef CGwmSpatialWeight spatial_weight = self._c_instance.spatialWeight()
        return (<CGwmBandwidthWeight*>(spatial_weight.weight())).bandwidth()
    
    def indep_vars(self):
        cdef CGwmSimpleLayer* layer = self._c_instance.resultLayer()
        cdef mat betas = self._c_instance.betas()
        return name_vector2list(layer.fields())[1:betas.n_cols]
    
    @property
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = self._c_instance.resultLayer()
        return CySimpleLayer(mat2numpy(layer.points()), 
                             mat2numpy(layer.data()),
                             name_vector2list(layer.fields()))


cdef class CyGWSS:
    cdef CGwmGWSS* _c_instance

    def __cinit__(self, CySimpleLayer layer, CyVariableList variable_list, CyWeight weight, CyDistance distance, bint quantile, bint first_only):
        self._c_instance = new CGwmGWSS()
        self._c_instance.setSourceLayer(layer._c_instance)
        self._c_instance.setVariables(variable_list._c_instance)
        cdef CGwmSpatialWeight spatial = CGwmSpatialWeight(weight._c_instance, distance._c_instance)
        self._c_instance.setSpatialWeight(spatial)
        self._c_instance.setQuantile(quantile)
        self._c_instance.setIsCorrWithFirstOnly(first_only)
    
    def enable_openmp(self, int threads):
        self._c_instance.setParallelType(ParallelType.OpenMP)
        self._c_instance.setOmpThreadNum(threads)
    
    def valid(self):
        return self._c_instance.isValid()
    
    def run(self):
        self._c_instance.run()

    def local_mean(self):
        return mat2numpy(self._c_instance.localMean())

    def local_sdev(self):
        return mat2numpy(self._c_instance.localSDev())

    def local_skewness(self):
        return mat2numpy(self._c_instance.localSkewness())

    def local_cv(self):
        return mat2numpy(self._c_instance.localCV())

    def local_var(self):
        return mat2numpy(self._c_instance.localVar())

    def local_median(self):
        return mat2numpy(self._c_instance.localMedian())

    def iqr(self):
        return mat2numpy(self._c_instance.iqr())

    def qi(self):
        return mat2numpy(self._c_instance.qi())

    def local_cov(self):
        return mat2numpy(self._c_instance.localCov())

    def local_corr(self):
        return mat2numpy(self._c_instance.localCorr())

    def local_scorr(self):
        return mat2numpy(self._c_instance.localSCorr())
    
    @property
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = self._c_instance.resultLayer()
        return CySimpleLayer(mat2numpy(layer.points()), 
                             mat2numpy(layer.data()),
                             name_vector2list(layer.fields()))


cdef class CyGWPCA:
    cdef CGwmGWPCA* _c_instance
    
    def __cinit__(self, CySimpleLayer layer, CyVariableList variable_list, CyWeight weight, CyDistance distance, int keepComponents):
        self._c_instance = new CGwmGWPCA()
        self._c_instance.setSourceLayer(layer._c_instance)
        self._c_instance.setVariables(variable_list._c_instance)
        cdef CGwmSpatialWeight spatial = CGwmSpatialWeight(weight._c_instance, distance._c_instance)
        self._c_instance.setSpatialWeight(spatial)
        self._c_instance.setKeepComponents(keepComponents)
    
    def valid(self):
        return self._c_instance.isValid()
    
    def run(self):
        self._c_instance.run()
    
    def local_pv(self):
        return mat2numpy(self._c_instance.localPV())

    def sdev(self):
        return mat2numpy(self._c_instance.sdev())
    
    def loadings(self):
        return cube2numpy(self._c_instance.loadings())

    def scores(self):
        return cube2numpy(self._c_instance.scores())
    
    @property
    def result_layer(self):
        cdef CGwmSimpleLayer* layer = self._c_instance.resultLayer()
        return CySimpleLayer(mat2numpy(layer.points()), 
                             mat2numpy(layer.data()),
                             name_vector2list(layer.fields()))
        
