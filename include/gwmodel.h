/**
 * @file gwmodel.h
 * @author YigongHu (hu_yigong@whu.edu.cn)
 * @brief This file provide the headers of this library. 
 * If the library is built as a static library, this header includes all the C++ headers. 
 * If the library is built as a shared library, this header provides interface functions for the calling of C++ classes.
 * @version 0.1.0
 * @date 2020-10-08
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef GWMODEL_H
#define GWMODEL_H

#ifndef GWMODEL_SHARED_LIB

#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmDistance.h"
#include "gwmodelpp/spatialweight/CGwmDMatDistance.h"
#include "gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "gwmodelpp/spatialweight/CGwmWeight.h"

#include "gwmodelpp/CGwmAlgorithm.h"
#include "gwmodelpp/CGwmBandwidthSelector.h"
#include "gwmodelpp/CGwmVariableForwardSelector.h"
#include "gwmodelpp/CGwmSimpleLayer.h"
#include "gwmodelpp/CGwmSpatialAlgorithm.h"
#include "gwmodelpp/CGwmSpatialMonoscaleAlgorithm.h"
#include "gwmodelpp/CGwmGWRBase.h"
#include "gwmodelpp/CGwmGWRBasic.h"

#else   // GWMODEL_SHARED_LIB

#ifdef WIN32
#ifdef CREATE_DLL_EXPORTS
#define GWMODEL_API __declspec(dllexport)
#else
#define GWMODEL_API __declspec(dllimport)
#endif 
#else
#define GWMODEL_API  
#endif

#include <cstddef>

#include "gwmodelpp/GwmRegressionDiagnostic.h"

/**
 * @brief An enum 
 * indicating the type of kernel functions
 */
enum KernelFunctionType
{
    Gaussian,       ///< Gaussian kernel
    Exponential,    ///< Exponential kernel
    Bisquare,       ///< Bisquare kernel
    Tricube,        ///< Tricube kernel
    Boxcar          ///< Boxcar kernel
};

/**
 * @brief An enum 
 * indicating the type of criterion using in bandwidth selection process
 */
enum BandwidthSelectionCriterionType
{
    AIC,    ///< use AIC as criterion
    CV      ///< use CV as criterion
};

class CGwmDistance;
class CGwmWeight;
class CGwmSpatialWeight;
class CGwmSimpleLayer;
class CGwmAlgorithm;
class CGwmSpatialAlgorithm;
class CGwmSpatialMonoscaleAlgorithm;
class CGwmGWRBase;
class CGwmGWRBasic;
class CGwmGWSS;


/**
 * @brief 
 * A very basic and simple struct representing a matrix, 
 * Usually is converted to an armadillo mat type in calculation, 
 * and some result matrices are converted to this type.
 */
extern "C" struct GwmMatInterface
{
    unsigned long long rows = 0;    ///< The number of rows in the matrix
    unsigned long long cols = 0;    ///< The number of columns in the matrix
    const double* data = nullptr;   ///< The pointer to data, usually created by `new` operator. Need to be deleted by the user.
};


/**
 * @brief A function to delete an instance of GwmMatInterface.
 * 
 * @param interface The pointer to the instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_mat(GwmMatInterface* interface);

/**
 * @brief A type used by names in different type. This type can contains only a 255-characters string.
 */
typedef char GwmNameInterface[256];


/**
 * @brief A struct representing a name list. 
 * Usually its instance is converted to a vector<string> instance in calculating,
 * and some result list are converted to this type.
 */
extern "C" struct GwmNameListInterface
{
    size_t size = 0;
    GwmNameInterface* items = nullptr;
};

/**
 * @brief A function to delete an instance of GwmNameListInterface.
 * 
 * @param interface The pointer to the instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_string_list(GwmNameListInterface* interface);


/**
 * @brief A struct of representing a variable. 
 * Usually its instance is converted to a GwmVariable instance in calculating,
 * and some return values are converted to this type.
 */
extern "C" struct GwmVariableInterface
{
    int index;
    bool isNumeric;
    GwmNameInterface name = "";
};


/**
 * @brief A struct representing a variable list. 
 * Usually its instance is converted to a vector<GwmVariable> instance in calculating,
 * and some return values are converted to this type.
 */
extern "C" struct GwmVariableListInterface
{
    size_t size = 0;
    GwmVariableInterface* items = nullptr;
};

/**
 * @brief A function to delete an instance of GwmVariableListInterface.
 * 
 * @param interface The pointer to the instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_variable_list(GwmVariableListInterface* interface);


/**
 * @brief A struct representing spatial bandwidth weight. 
 * Usually its instance is converted to a GwmBandwidthWeight instance in calculating,
 * and some return values are converted to this type.
 */
extern "C" struct GwmBandwidthKernelInterface
{
    double size;
    bool isAdaptive;
    KernelFunctionType type;
};


/**
 * @brief A struct representing a pair of bandwidth and its criterion.
 */
extern "C" struct GwmBandwidthCriterionPairInterface
{
    double bandwidth;
    double criterion;
};


/**
 * @brief A struct representing a list of GwmBandwidthCriterionPairInterface.
 */
extern "C" struct GwmBandwidthCriterionListInterface
{
    size_t size = 0;
    GwmBandwidthCriterionPairInterface* items = nullptr;
};

/**
 * @brief A function to delete an instance of GwmBandwidthCriterionListInterface.
 * 
 * @param interface The pointer to the instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_bandwidth_criterion_list(GwmBandwidthCriterionListInterface* interface);


/**
 * @brief A struct representing a pair of variables and its criterion.
 */
extern "C" struct GwmVariablesCriterionPairInterface
{
    GwmVariableListInterface variables;
    double criterion;
};

/**
 * @brief A function to delete an instance of GwmVariablesCriterionPairInterface.
 * 
 * @param interface The pointer to the instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_variable_criterion_pair(GwmVariablesCriterionPairInterface* interface);

/**
 * @brief A struct representing a list of GwmVariablesCriterionPairInterface.
 */
extern "C" struct GwmVariablesCriterionListInterface
{
    size_t size = 0;
    GwmVariablesCriterionPairInterface* items = nullptr;
};

/**
 * @brief A function to delete an instance of GwmVariablesCriterionListInterface.
 * 
 * @param interface The pointer to the instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_variable_criterion_list(GwmVariablesCriterionListInterface* interface);

/**
 * @brief A function to create a CGwmCRSDistance instance.
 * 
 * @param isGeogrphical Set to true if the coordinates of data is longitude/latitude. 
 * @return CGwmDistance* A base class CGwmDistance pointer to be used in CGwmSpatialWeight.
 */
extern "C" GWMODEL_API CGwmDistance* gwmodel_create_crs_distance(bool isGeogrphical);

/**
 * @brief A function to create a CGwmBandwidthWeight instance. 
 * 
 * @param size The bandwidth size. 
 * @param isAdaptive Set to true if use an adaptive bandwidth.
 * @param type Set the type of kernel function. 
 * @return CGwmWeight* A base class CGwmWeight pointer to be used in CGwmSpatialWeight.
 */
extern "C" GWMODEL_API CGwmWeight* gwmodel_create_bandwidth_weight(double size, bool isAdaptive, KernelFunctionType type);

/**
 * @brief A function to create a CGwmBandwidthWeight instance.
 * 
 * @param distance The CGwmDistance pointer created by gwmodel_create_crs_distance() .
 * @param weight The CGwmWeight pointer created by gwmodel_create_bandwidth_weight() .
 * @return CGwmSpatialWeight* A pointer to the CGwmSpatialWeight instance created by this function.
 */
extern "C" GWMODEL_API CGwmSpatialWeight* gwmodel_create_spatial_weight(CGwmDistance* distance, CGwmWeight* weight);

/**
 * @brief A function to create a CGwmSimpleLayer instance.
 * 
 * @param pointsInterface A mat containing coordinates of points. 
 * The shape of it must be nx2 and the first column is longitudes or x-coordinate, the second column is latitudes or y-coordinate.
 * @param dataInterface A mat containing data. The number of rows of must be equal to that of pointsInterface, and the number of columns must be equal to that of fieldsInterface.
 * @param fieldsInterface A list containing names of fields.
 * @return CGwmSimpleLayer* A pointer to the CGwmSimpleLayer instance created by this function.
 */
extern "C" GWMODEL_API CGwmSimpleLayer* gwmodel_create_simple_layer(GwmMatInterface pointsInterface, GwmMatInterface dataInterface, GwmNameListInterface fieldsInterface);

/**
 * @deprecated This function has no effect. It always return a nullptr.
 */
extern "C" GWMODEL_API CGwmSpatialAlgorithm* gwmodel_create_algorithm();

/**
 * @brief A function to create a CGwmGWRBasic instance. This function create only a initialized instance, and is usually inexecutable. 
 * Users need to set some properties of it.
 * 
 * @return CGwmGWRBasic* A pointer to the CGwmGWRBasic instance created by this function.
 */
extern "C" GWMODEL_API CGwmGWRBasic* gwmodel_create_gwr_algorithm();

/**
 * @brief A function to create a CGwmGWSS instance. This function create only a initialized instance, and is usually inexecutable. 
 * Users need to set some properties of it.
 * 
 * @return CGwmGWSS* A pointer to the CGwmGWSS instance created by this function.
 */
extern "C" GWMODEL_API CGwmGWSS* gwmodel_create_gwss_algorithm();


/**
 * @brief A function to delete CGwmDistance instance.
 * 
 * @param instance Pointer to the CGwmDistance instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_crs_distance(CGwmDistance* instance);

/**
 * @brief A function to delete CGwmWeight instance.
 * 
 * @param instance Pointer to the CGwmWeight instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_bandwidth_weight(CGwmWeight* instance);

/**
 * @brief A function to delete CGwmSpatialWeight instance.
 * 
 * @param instance Pointer to the CGwmSpatialWeight instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_spatial_weight(CGwmSpatialWeight* instance);

/**
 * @brief A function to delete CGwmSimpleLayer instance.
 * 
 * @param instance Pointer to the CGwmSimpleLayer instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_simple_layer(CGwmSimpleLayer* instance);

/**
 * @brief A function to delete CGwmSpatialAlgorithm instance.
 * 
 * @param instance Pointer to the CGwmSpatialAlgorithm instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_algorithm(CGwmSpatialAlgorithm* instance);

/**
 * @brief A function to delete CGwmGWRBasic instance.
 * 
 * @param instance Pointer to the CGwmGWRBasic instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_gwr_algorithm(CGwmGWRBasic* instance);

/**
 * @brief A function to delete CGwmGWSS instance.
 * 
 * @param instance Pointer to the CGwmGWSS instance.
 */
extern "C" GWMODEL_API void gwmodel_delete_gwss_algorithm(CGwmGWSS* instance);

/**
 * @brief Set data source layer for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param layer Pointer to data source layer returned from gwmodel_create_simple_layer().
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_source_layer(CGwmGWRBasic* algorithm, CGwmSimpleLayer* layer);

/**
 * @brief Set spatial weight configuration for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param spatial Pointer returned from gwmodel_create_spatial_weight().
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_spatial_weight(CGwmGWRBasic* algorithm, CGwmSpatialWeight* spatial);

/**
 * @brief Set dependent variables for CGwmGWRBasic algorithm.
 * 
 * @param regression Pointer returned from gwmodel_create_gwr_algorithm().
 * @param depVar Dependent variable.
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_dependent_variable(CGwmGWRBasic* regression, GwmVariableInterface depVar);

/**
 * @brief Set independent variable list for CGwmGWRBasic algorithm.
 * 
 * @param regression Pointer returned from gwmodel_create_gwr_algorithm().
 * @param indepVarList Independent variable list.
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_independent_variable(CGwmGWRBasic* regression, GwmVariableListInterface indepVarList);

/**
 * @brief Set prediction layer for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param layer Pointer to prediction layer returned from gwmodel_create_simple_layer().
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_predict_layer(CGwmGWRBasic* algorithm, CGwmSimpleLayer* layer);

/**
 * @brief Enable bandwidth autoselection and set criterion for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param criterion Type of bandwidth criterion.
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_bandwidth_autoselection(CGwmGWRBasic* algorithm, BandwidthSelectionCriterionType criterion);

/**
 * @brief Enable independent-variables autoselection and set threshold for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param threshold Threshold used in the autoselection process.
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_indep_vars_autoselection(CGwmGWRBasic* algorithm, double threshold);

/**
 * @brief Set some other options for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param hasHatMatrix Whether calculate a hat matrix which used to calculate diagnostic informations. 
 * If prediction layer is set, this configure has no effects. 
 * If only data source layer is set, set this configure to false to disable diagnostic.
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_options(CGwmGWRBasic* algorithm, bool hasHatMatrix);

/**
 * @brief Enable multi-thread parallelized algorithm and set number of concurrent threads for CGwmGWRBasic algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 * @param threads Number of concurrent threads.
 */
extern "C" GWMODEL_API void gwmodel_set_gwr_openmp(CGwmGWRBasic* algorithm, int threads);


/**
 * @brief Set data source layer for CGwmGWSS algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwss_algorithm().
 * @param layer Pointer to data source layer returned from gwmodel_create_simple_layer().
 */
extern "C" GWMODEL_API void gwmodel_set_gwss_source_layer(CGwmGWSS* algorithm, CGwmSimpleLayer* layer);

/**
 * @brief Set spatial weight configuration for CGwmGWSS algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwss_algorithm().
 * @param spatial Pointer returned from gwmodel_create_spatial_weight().
 */
extern "C" GWMODEL_API void gwmodel_set_gwss_spatial_weight(CGwmGWSS* algorithm, CGwmSpatialWeight* spatial);

/**
 * @brief Set variable list for CGwmGWSS algorithm.
 * 
 * @param regression Pointer returned from gwmodel_create_gwss_algorithm().
 * @param indepVarList Independent variable list.
 */
extern "C" GWMODEL_API void gwmodel_set_gwss_variables(CGwmGWSS* algorithm, GwmVariableListInterface varList);

/**
 * @brief Set some other options for CGwmGWSS algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwss_algorithm().
 * @param quantile Whether median, interquartile range, quantile imbalance will be calculated. True to be calculated. 
 * @param corrWithFirstOnly Whether only local covariances, local correlations (Pearson's) and local correlations (Spearman's) 
 * between the first variable and the other variables are calculated.
 */
extern "C" GWMODEL_API void gwmodel_set_gwss_options(CGwmGWSS* algorithm, bool quantile, bool corrWithFirstOnly);

/**
 * @brief Enable multi-thread parallelized algorithm and set number of concurrent threads for CGwmGWSS algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwss_algorithm().
 * @param threads Number of concurrent threads.
 */
extern "C" GWMODEL_API void gwmodel_set_gwss_openmp(CGwmGWSS* algorithm, int threads);

/**
 * @brief Run basic GWR algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwr_algorithm().
 */
extern "C" GWMODEL_API void gwmodel_run_gwr(CGwmGWRBasic* algorithm);

/**
 * @brief Run GWSS algorithm.
 * 
 * @param algorithm Pointer returned from gwmodel_create_gwss_algorithm()
 */
extern "C" GWMODEL_API void gwmodel_run_gwss(CGwmGWSS* algorithm);


/**
 * @brief Get coordinates of points from layer.
 * 
 * @param layer Pointer to layer.
 * @return GwmMatInterface Mat struct containing coordinates of the features.
 * The shape of it is nx2 and the first column is longitudes or x-coordinate, the second column is latitude or y-coordinate.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_simple_layer_points(CGwmSimpleLayer* layer);

/**
 * @brief Get data from layer.
 * 
 * @param layer Pointer to layer.
 * @return GwmMatInterface Mat struct containing data of the features.
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_simple_layer_data(CGwmSimpleLayer* layer);

/**
 * @brief Get fields from layer.
 * 
 * @param layer Pointer to layer.
 * @return GwmNameListInterface A list containing names of fields.
 */
extern "C" GWMODEL_API GwmNameListInterface gwmodel_get_simple_layer_fields(CGwmSimpleLayer* layer);

/**
 * @brief Get spatial weight configuration from GWR algorithm.
 * 
 * @param gwr Pointer to GWR algorithm.
 * @return CGwmSpatialWeight* Pointer to spatial weight configuration instance.
 */
extern "C" GWMODEL_API CGwmSpatialWeight* gwmodel_get_gwr_spatial_weight(CGwmGWRBasic* gwr);

/**
 * @brief Get result layer from GWR algorithm.
 * 
 * @param gwr Pointer to GWR algorithm.
 * @return CGwmSimpleLayer* Pointer to result layer.
 */
extern "C" GWMODEL_API CGwmSimpleLayer* gwmodel_get_gwr_result_layer(CGwmGWRBasic* gwr);

/**
 * @brief Get regression coefficients from GWR algorithm.
 * 
 * @param gwr Pointer to GWR algorithm.
 * @return GwmMatInterface Mat struct of coefficients. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwr_coefficients(CGwmGWRBasic* gwr);

/**
 * @brief Get independent variables and its criterions tested in variable autoselection from GWR algorithm.
 * 
 * @param gwr Pointer to GWR algorithm.
 * @return GwmVariablesCriterionListInterface Struct of independent variables and its criterions.
 */
extern "C" GWMODEL_API GwmBandwidthCriterionListInterface gwmodel_get_gwr_bandwidth_criterions(CGwmGWRBasic* gwr);

/**
 * @brief Get independent variables and its criterions tested in variable autoselection from GWR algorithm.
 * 
 * @param gwr Pointer to GWR algorithm.
 * @return GwmVariablesCriterionListInterface Struct of independent variables and its criterions.
 */
extern "C" GWMODEL_API GwmVariablesCriterionListInterface gwmodel_get_gwr_indep_var_criterions(CGwmGWRBasic* gwr);

/**
 * @brief Get diagnostic information from GWR algorithm.
 * 
 * @param gwr Pointer to GWR algorithm.
 * @return GwmRegressionDiagnostic Struct of diagnostic information
 */
extern "C" GWMODEL_API GwmRegressionDiagnostic gwmodel_get_gwr_diagnostic(CGwmGWRBasic* gwr);

/**
 * @brief Get result layer from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return CGwmSimpleLayer* Pointer to result layer.
 */
extern "C" GWMODEL_API CGwmSimpleLayer* gwmodel_get_gwss_result_layer(CGwmGWSS* gwss);

/**
 * @brief Get local mean from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local mean. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_mean(CGwmGWSS* gwss);

/**
 * @brief Get local standard deviations from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local standard deviations. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_sdev(CGwmGWSS* gwss);

/**
 * @brief Get local variance from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local variance. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_var(CGwmGWSS* gwss);

/**
 * @brief Get local skew from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local skew. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_skew(CGwmGWSS* gwss);

/**
 * @brief Get local coefficients of variation from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local coefficients of variation. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_cv(CGwmGWSS* gwss);

/**
 * @brief Get local covariances from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local covariances. 
 * The number of rows is the same as number of features. 
 * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
 * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
 * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
 * \f$cov(v_1,v_2), cov(v_1,v_3), ... , cov(v_1,v_k), cov(v_2,v_3), ... , cov(v_2,v_k), ... , cov(v_{k-1},vk)\f$
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_cov(CGwmGWSS* gwss);

/**
 * @brief Get local correlations (Pearson's) from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local correlations (Pearson's). 
 * The number of rows is the same as number of features. 
 * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
 * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
 * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
 * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_corr(CGwmGWSS* gwss);

/**
 * @brief Get local correlations (Spearman's), from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local correlations (Spearman's),. 
 * The number of rows is the same as number of features. 
 * If corrWithFirstOnly is set true, the number of columns is the (number of fields) - 1;
 * if not, the number of columns is the (((number of fields) - 1) * (number of fields)) / 2.
 * For variables \f$v_1, v_2, v_3, ... , v_{k-1}, v_k\f$, the fields are arranged as: 
 * \f$corr(v_1,v_2), corr(v_1,v_3), ... , corr(v_1,v_k), corr(v_2,v_3), ... , corr(v_2,v_k), ... , corr(v_{k-1},vk)\f$
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_spearman_rho(CGwmGWSS* gwss);

/**
 * @brief Get local medians from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local medians. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_median(CGwmGWSS* gwss);

/**
 * @brief Get local interquartile ranges from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local interquartile ranges. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_iqr(CGwmGWSS* gwss);

/**
 * @brief Get local quantile imbalances from GWSS algorithm.
 * 
 * @param gwss Pointer to GWSS algorithm.
 * @return GwmMatInterface Mat struct of local quantile imbalances. 
 * The number of rows is the same as number of features. 
 * The number of columns is the same as number of fields.
 */
extern "C" GWMODEL_API GwmMatInterface gwmodel_get_gwss_local_qi(CGwmGWSS* gwss);

/**
 * @brief Get spatial bandwidth weight from CGwmSpatialWeight.
 * 
 * @param spatial Pointer to spaital weight configuration.
 * @param bandwidth Pointer to store bandwidth weight.
 * @return true The spatial weight contains a CGwmBandwidthWeight weight member and bandwidth is correctly set.
 * @return false The spatial weight don't contain a CGwmBandwidthWeight weight member
 */
extern "C" GWMODEL_API bool gwmodel_get_spatial_bandwidth_weight(CGwmSpatialWeight* spatial, GwmBandwidthKernelInterface* bandwidth);

/**
 * @brief Convert a CGwmWeight to CGwmBandwidthWeight
 * 
 * @param weight Pointer to weight instance.
 * @param bandwidth Pointer to store bandwidth weight.
 * @return true The pointer can be threated as a CGwmBandwidthWeight pointer and bandwidth is correctly set.
 * @return false The pointer cannot be threated as a CGwmBandwidthWeight pointer.
 */
extern "C" GWMODEL_API bool gwmodel_as_bandwidth_weight(CGwmWeight* weight, GwmBandwidthKernelInterface* bandwidth);

#endif

#endif  // GWMODEL_H