#include "gwmodel.h"

#include <assert.h>
#include <vector>
#include <armadillo>
#include "GwmVariable.h"
#include "spatialweight/CGwmCRSDistance.h"
#include "spatialweight/CGwmBandwidthWeight.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "CGwmSimpleLayer.h"
#include "CGwmSpatialAlgorithm.h"
#include "CGwmSpatialMonoscaleAlgorithm.h"
#include "CGwmGWRBase.h"
#include "CGwmGWRBasic.h"
#include "CGwmGWSS.h"

using namespace std;
using namespace arma;

GwmMatInterface mat2interface(const mat& armamat)
{
    GwmMatInterface interface;
    interface.cols = armamat.n_cols;
    interface.rows = armamat.n_rows;
    interface.data = new double[armamat.n_elem];
    memcpy((void*)interface.data, armamat.memptr(), armamat.n_elem * sizeof(double));
    return interface;
}

void gwmodel_delete_mat(GwmMatInterface* interface)
{
    if (interface->data) delete[] interface->data;
    interface->data = nullptr;
    interface->cols = 0;
    interface->rows = 0;
}

void gwmodel_delete_string_list(GwmNameListInterface* interface)
{
    if (interface->items) delete[] interface->items;
    interface->items = nullptr;
    interface->size = 0;
}

void gwmodel_delete_variable_list(GwmVariableListInterface* interface)
{
    if (interface->items) delete[] interface->items;
    interface->items = nullptr;
    interface->size = 0;
}

void gwmodel_delete_bandwidth_criterion_list(GwmBandwidthCriterionListInterface* interface)
{
    if (interface->items) delete[] interface->items;
    interface->items = nullptr; 
    interface->size = 0;
}

void gwmodel_delete_variable_criterion_pair(GwmVariablesCriterionPairInterface* interface)
{
    gwmodel_delete_variable_list(&interface->variables);
    interface->criterion = 0.0;
}

void gwmodel_delete_variable_criterion_list(GwmVariablesCriterionListInterface* interface)
{
    if (interface->items)
    {
        for (size_t i = 0; i < interface->size; i++)
        {
            gwmodel_delete_variable_criterion_pair(interface->items + i);
        }
        delete[] interface->items;
        interface->items = nullptr;
    }
    interface->size = 0;
}

CGwmSpatialAlgorithm* gwmodel_create_algorithm()
{
    return nullptr;
}

CGwmDistance* gwmodel_create_crs_distance(bool isGeogrphical)
{
    return new CGwmCRSDistance(isGeogrphical);
}

CGwmWeight* gwmodel_create_bandwidth_weight(double size, bool isAdaptive, KernelFunctionType type)
{
    return new CGwmBandwidthWeight(size, isAdaptive, (CGwmBandwidthWeight::KernelFunctionType)type);
}

CGwmSpatialWeight* gwmodel_create_spatial_weight(CGwmDistance* distance, CGwmWeight* weight)
{
    return new CGwmSpatialWeight(weight, distance);
}

CGwmSimpleLayer* gwmodel_create_simple_layer(GwmMatInterface pointsInterface, GwmMatInterface dataInterface, GwmNameListInterface fieldsInterface)
{
    mat points(pointsInterface.data, pointsInterface.rows, pointsInterface.cols);
    mat data(dataInterface.data, dataInterface.rows, dataInterface.cols);
    vector<string> fields;
    for (size_t i = 0; i < fieldsInterface.size; i++)
    {
        fields.push_back(string(fieldsInterface.items[i]));
    }
    return new CGwmSimpleLayer(points, data, fields);
}

CGwmGWRBasic* gwmodel_create_gwr_algorithm()
{
    CGwmGWRBasic* algorithm = new CGwmGWRBasic();
    return algorithm;
}

CGwmGWSS* gwmodel_create_gwss_algorithm()
{
    return new CGwmGWSS();
}

void gwmodel_delete_crs_distance(CGwmDistance* instance)
{
    delete instance;
}

void gwmodel_delete_bandwidth_weight(CGwmWeight* instance)
{
    delete instance;
}

void gwmodel_delete_spatial_weight(CGwmSpatialWeight* instance)
{
    delete instance;
}

void gwmodel_delete_simple_layer(CGwmSimpleLayer* instance)
{
    delete instance;
}

void gwmodel_delete_algorithm(CGwmSpatialAlgorithm* instance)
{
    delete instance;
}

void gwmodel_delete_gwr_algorithm(CGwmGWRBasic* instance)
{
    delete instance;
}

void gwmodel_delete_gwss_algorithm(CGwmGWSS* instance)
{
    delete instance;
}

void gwmodel_set_gwr_source_layer(CGwmGWRBasic* algorithm, CGwmSimpleLayer* layer)
{
    algorithm->setSourceLayer(layer);
}

void gwmodel_set_gwr_predict_layer(CGwmGWRBasic* algorithm, CGwmSimpleLayer* layer)
{
    algorithm->setPredictLayer(layer);
}

void gwmodel_set_gwr_dependent_variable(CGwmGWRBasic* regression, GwmVariableInterface depVar)
{
    regression->setDependentVariable(GwmVariable(depVar.index, depVar.isNumeric, depVar.name));
}

void gwmodel_set_gwr_independent_variable(CGwmGWRBasic* regression, GwmVariableListInterface indepVarList)
{
    vector<GwmVariable> indepVars;
    for (size_t i = 0; i < indepVarList.size; i++)
    {
        GwmVariableInterface* vi = indepVarList.items + i;
        assert(vi);
        GwmVariable v(vi->index, vi->isNumeric, vi->name);
        indepVars.push_back(v);
    }
    regression->setIndependentVariables(indepVars);
}

void gwmodel_set_gwr_spatial_weight(CGwmGWRBasic* algorithm, CGwmSpatialWeight* spatial)
{
    algorithm->setSpatialWeight(*spatial);
}

void gwmodel_set_gwr_bandwidth_autoselection(CGwmGWRBasic* algorithm, BandwidthSelectionCriterionType criterion)
{
    algorithm->setIsAutoselectBandwidth(true);
    algorithm->setBandwidthSelectionCriterion((CGwmGWRBasic::BandwidthSelectionCriterionType)criterion);
}

void gwmodel_set_gwr_indep_vars_autoselection(CGwmGWRBasic* algorithm, double threshold)
{
    algorithm->setIsAutoselectIndepVars(true);
    algorithm->setIndepVarSelectionThreshold(threshold);
}

void gwmodel_set_gwr_options(CGwmGWRBasic* algorithm, bool hasHatMatrix)
{
    algorithm->setHasHatMatrix(hasHatMatrix);
}

void gwmodel_set_gwr_openmp(CGwmGWRBasic* algorithm, int threads)
{
    algorithm->setParallelType(ParallelType::OpenMP);
    algorithm->setOmpThreadNum(threads);
}

void gwmodel_set_gwss_source_layer(CGwmGWSS* algorithm, CGwmSimpleLayer* layer)
{
    algorithm->setSourceLayer(layer);
}

void gwmodel_set_gwss_spatial_weight(CGwmGWSS* algorithm, CGwmSpatialWeight* spatial)
{
    algorithm->setSpatialWeight(*spatial);
}

void gwmodel_set_gwss_variables(CGwmGWSS* algorithm, GwmVariableListInterface varList)
{
    vector<GwmVariable> vars;
    for (size_t i = 0; i < varList.size; i++)
    {
        GwmVariableInterface* vi = varList.items + i;
        assert(vi);
        GwmVariable v(vi->index, vi->isNumeric, vi->name);
        vars.push_back(v);
    }
    algorithm->setVariables(vars);
}

void gwmodel_set_gwss_options(CGwmGWSS* algorithm, bool quantile, bool corrWithFirstOnly)
{
    algorithm->setQuantile(quantile);
    algorithm->setIsCorrWithFirstOnly(corrWithFirstOnly);
}

void gwmodel_set_gwss_openmp(CGwmGWSS* algorithm, int threads)
{
    algorithm->setParallelType(ParallelType::OpenMP);
    algorithm->setOmpThreadNum(threads);
}

void gwmodel_run_gwr(CGwmGWRBasic* algorithm)
{
    algorithm->run();
}

void gwmodel_run_gwss(CGwmGWSS* algorithm)
{
    algorithm->run();
}

GwmMatInterface gwmodel_get_simple_layer_points(CGwmSimpleLayer* layer)
{
    return mat2interface(layer->points());
}

GwmMatInterface gwmodel_get_simple_layer_data(CGwmSimpleLayer* layer)
{
    return mat2interface(layer->data());
}

GwmNameListInterface gwmodel_get_simple_layer_fields(CGwmSimpleLayer* layer)
{
    vector<string> fields = layer->fields();
    GwmNameListInterface fieldsInterface;
    fieldsInterface.size = fields.size();
    fieldsInterface.items = new GwmNameInterface[fieldsInterface.size];
    for (size_t i = 0; i < fieldsInterface.size; i++)
    {
        string f = fields[i];
        strcpy(fieldsInterface.items[i], f.data());
    }
    return fieldsInterface;
}

CGwmSpatialWeight* gwmodel_get_gwr_spatial_weight(CGwmGWRBasic* gwr)
{
    return new CGwmSpatialWeight(gwr->spatialWeight());
}

CGwmSimpleLayer* gwmodel_get_gwr_result_layer(CGwmGWRBasic* gwr)
{
    return gwr->resultLayer();
}

GwmMatInterface gwmodel_get_gwr_coefficients(CGwmGWRBasic* gwr)
{
    return mat2interface(gwr->betas());
}

GwmRegressionDiagnostic gwmodel_get_gwr_diagnostic(CGwmGWRBasic* gwr)
{
    return gwr->diagnostic();
}

GwmBandwidthCriterionListInterface gwmodel_get_gwr_bandwidth_criterions(CGwmGWRBasic* gwr)
{
    BandwidthCriterionList criterions = gwr->bandwidthSelectionCriterionList();
    GwmBandwidthCriterionListInterface interface;
    interface.size = criterions.size();
    interface.items = new GwmBandwidthCriterionPairInterface[interface.size];
    for (size_t i = 0; i < interface.size; i++)
    {
        GwmBandwidthCriterionPairInterface* item = interface.items + i;
        item->bandwidth = criterions[i].first;
        item->criterion = criterions[i].second;
    }
    return interface;
}

GwmVariablesCriterionListInterface gwmodel_get_gwr_indep_var_criterions(CGwmGWRBasic* gwr)
{
    VariablesCriterionList criterions = gwr->indepVarsSelectionCriterionList();
    GwmVariablesCriterionListInterface interface;
    interface.size = criterions.size();
    interface.items = new GwmVariablesCriterionPairInterface[interface.size];
    for (size_t i = 0; i < interface.size; i++)
    {
        GwmVariablesCriterionPairInterface* item = interface.items + i;
        item->criterion = criterions[i].second;
        vector<GwmVariable> varList = criterions[i].first;
        item->variables.size = varList.size();
        item->variables.items = new GwmVariableInterface[item->variables.size];
        for (size_t v = 0; v < item->variables.size; v++)
        {
            GwmVariableInterface* vi = item->variables.items + v;
            vi->index = varList[v].index;
            vi->isNumeric = varList[v].isNumeric;
            strcpy(vi->name, varList[v].name.data());
        }
    }
    return interface;
}

CGwmSimpleLayer* gwmodel_get_gwss_result_layer(CGwmGWSS* gwss)
{
    return gwss->resultLayer();
}

GwmMatInterface gwmodel_get_gwss_local_mean(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localMean());
}

GwmMatInterface gwmodel_get_gwss_local_sdev(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localSDev());
}

GwmMatInterface gwmodel_get_gwss_local_var(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localVar());
}

GwmMatInterface gwmodel_get_gwss_local_skew(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localSkewness());
}

GwmMatInterface gwmodel_get_gwss_local_cv(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localCV());
}

GwmMatInterface gwmodel_get_gwss_local_cov(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localCov());
}

GwmMatInterface gwmodel_get_gwss_local_corr(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localCorr());
}

GwmMatInterface gwmodel_get_gwss_local_spearman_rho(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localSCorr());
}

GwmMatInterface gwmodel_get_gwss_local_median(CGwmGWSS* gwss)
{
    return mat2interface(gwss->localMedian());
}

GwmMatInterface gwmodel_get_gwss_local_iqr(CGwmGWSS* gwss)
{
    return mat2interface(gwss->iqr());
}

GwmMatInterface gwmodel_get_gwss_local_qi(CGwmGWSS* gwss)
{
    return mat2interface(gwss->qi());
}

bool gwmodel_as_bandwidth_weight(CGwmWeight* weight, GwmBandwidthKernelInterface* bandwidth)
{
    CGwmBandwidthWeight* bw = dynamic_cast<CGwmBandwidthWeight*>(weight);
    if (bw)
    {
        *bandwidth = { bw->bandwidth(), bw->adaptive(), (KernelFunctionType)bw->kernel() };
        return true;
    }
    else return false;
}

bool gwmodel_get_spatial_bandwidth_weight(CGwmSpatialWeight* spatial, GwmBandwidthKernelInterface* bandwidth)
{
    CGwmBandwidthWeight* bw = dynamic_cast<CGwmBandwidthWeight*>(spatial->weight());
    if (bw)
    {
        *bandwidth = { bw->bandwidth(), bw->adaptive(), (KernelFunctionType)bw->kernel() };
        return true;
    }
    else return false;
}