#include "GWRMultiscale.h"
#ifdef ENABLE_OPENMP
#include <omp.h>
#endif
#include <exception>
#include <vector>
#include <string>
#include <spatialweight/CRSDistance.h>
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include "Logger.h"
#include "GWRBasic.h"

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "cumat.hpp"
#include "CudaUtils.h"
#endif // ENABLE_CUDA

using namespace std;
using namespace arma;
using namespace gwm;

int GWRMultiscale::treeChildCount = 0;

unordered_map<GWRMultiscale::BandwidthInitilizeType,string> GWRMultiscale::BandwidthInitilizeTypeNameMapper = {
    make_pair(GWRMultiscale::BandwidthInitilizeType::Null, ("Not initilized, not specified")),
    make_pair(GWRMultiscale::BandwidthInitilizeType::Initial, ("Initilized")),
    make_pair(GWRMultiscale::BandwidthInitilizeType::Specified, ("Specified"))
};

unordered_map<GWRMultiscale::BandwidthSelectionCriterionType,string> GWRMultiscale::BandwidthSelectionCriterionTypeNameMapper = {
    make_pair(GWRMultiscale::BandwidthSelectionCriterionType::CV, ("CV")),
    make_pair(GWRMultiscale::BandwidthSelectionCriterionType::AIC, ("AIC"))
};

unordered_map<GWRMultiscale::BackFittingCriterionType,string> GWRMultiscale::BackFittingCriterionTypeNameMapper = {
    make_pair(GWRMultiscale::BackFittingCriterionType::CVR, ("CVR")),
    make_pair(GWRMultiscale::BackFittingCriterionType::dCVR, ("dCVR"))
};

RegressionDiagnostic GWRMultiscale::CalcDiagnostic(const mat &x, const vec &y, const vec &shat, double RSS)
{
    // 诊断信息
    double nDp = (double)x.n_rows;
    double RSSg = RSS;
    double sigmaHat21 = RSSg / nDp;
    double TSS = sum((y - mean(y)) % (y - mean(y)));
    double Rsquare = 1 - RSSg / TSS;

    double trS = shat(0);
    double trStS = shat(1);
    double enp = 2 * trS - trStS;
    double edf = nDp - 2 * trS + trStS;
    double AICc = nDp * log(sigmaHat21) + nDp * log(2 * M_PI) + nDp * ((nDp + trS) / (nDp - 2 - trS));
    double adjustRsquare = 1 - (1 - Rsquare) * (nDp - 1) / (edf - 1);

    // 保存结果
    RegressionDiagnostic diagnostic;
    diagnostic.RSS = RSSg;
    diagnostic.AICc = AICc;
    diagnostic.ENP = enp;
    diagnostic.EDF = edf;
    diagnostic.RSquareAdjust = adjustRsquare;
    diagnostic.RSquare = Rsquare;
    return diagnostic;
}

mat GWRMultiscale::fit()
{
    GWM_LOG_STAGE("Initializing");
    uword nDp = mX.n_rows, nVar = mX.n_cols;
    createDistanceParameter(nVar);
    createInitialDistanceParameter();
    mMaxDistances.resize(nVar);
#ifdef ENABLE_CUDA
    if (mParallelType & ParallelType::CUDA)
    {
        cublasCreate(&cubase::handle);
        mInitSpatialWeight.prepareCuda(mGpuId);
        for (size_t i = 0; i < nVar; i++)
        {
            mSpatialWeights[i].prepareCuda(mGpuId);
        }
    }
#endif // ENABLE_CUDA
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // ********************************
    // Centering and scaling predictors
    // ********************************
    mX0 = mX;
    mY0 = mY;
    for (uword i = (mHasIntercept ? 1 : 0); i < nVar ; i++)
    {
        if (mPreditorCentered[i])
        {
            mX.col(i) = mX.col(i) - mean(mX.col(i));
        }
    }
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // ***********************
    // Intialize the bandwidth
    // ***********************
    GWM_LOG_STAGE("Calculating initial bandwidths");
    mYi = mY;
    for (uword i = 0; i < nVar ; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        if (mBandwidthInitilize[i] == BandwidthInitilizeType::Null)
        {
            mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
            mBandwidthSelectionCurrentIndex = i;
            mXi = mX.col(i);
            BandwidthWeight* bw0 = bandwidth(i);
            bool adaptive = bw0->adaptive();
            mMaxDistances[i] = mSpatialWeights[i].distance()->maxDistance();

            GWM_LOG_INFO(string(GWM_LOG_TAG_MGWR_INITIAL_BW) + to_string(i));
            BandwidthSelector selector;
            selector.setBandwidth(bw0);
            selector.setLower(mGoldenLowerBounds.value_or(adaptive ? mAdaptiveLower : mMaxDistances[i] / 5000.0));
            selector.setUpper(mGoldenUpperBounds.value_or(adaptive ? mCoords.n_rows : mMaxDistances[i]));
            BandwidthWeight* bw = selector.optimize(this);
            if (bw)
            {
                mSpatialWeights[i].setWeight(bw);
#ifdef ENABLE_CUDA
                if (mParallelType & ParallelType::CUDA)
                    mSpatialWeights[i].prepareCuda(mGpuId);
#endif // ENABLE_CUDA
            }
            GWM_LOG_INFO(string(GWM_LOG_TAG_MGWR_INITIAL_BW)  + to_string(i)  + "," + to_string(bw->bandwidth()));
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    }

    // *****************************************************
    // Calculate the initial beta0 from the above bandwidths
    // *****************************************************
    GWM_LOG_STAGE("Calculating initial bandwidth");
    GWRBasic gwr;
    gwr.setCoords(mCoords);
    gwr.setDependentVariable(mY);
    gwr.setIndependentVariables(mX);
    gwr.setSpatialWeight(mSpatialWeights[0]);
    gwr.setIsAutoselectBandwidth(true);
    switch (mBandwidthSelectionApproach[0])
    {
    case GWRMultiscale::BandwidthSelectionCriterionType::CV:
        gwr.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
        break;
    case GWRMultiscale::BandwidthSelectionCriterionType::AIC:
        gwr.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
    default:
        break;
    }
    gwr.setParallelType(mParallelType);
    switch (mParallelType)
    {
    case ParallelType::OpenMP:
    case ParallelType::MPI_MP:
        gwr.setOmpThreadNum(mOmpThreadNum);
        break;
    case ParallelType::CUDA:
    case ParallelType::MPI_CUDA:
        gwr.setGroupSize(mGroupLength);
        gwr.setGPUId(mGpuId);
    default:
        break;
    }
    gwr.setStoreS(mHasHatMatrix);
    gwr.setStoreC(mHasHatMatrix);
    mat betas = gwr.fit();
    mBetasSE = gwr.betasSE();
    BandwidthWeight* initBw = gwr.spatialWeight().weight<BandwidthWeight>();
    if (!initBw)
    {
        throw std::runtime_error("Cannot select initial bandwidth.");
    }
    mInitSpatialWeight.setWeight(initBw);
#ifdef ENABLE_CUDA
    mInitSpatialWeight.prepareCuda(mGpuId);
#endif // ENABLE_CUDA
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // 初始化诊断信息矩阵
    if (mHasHatMatrix)
    {
        mS0 = gwr.s();
        mSArray = cube(nDp, nDp, nVar, fill::zeros);
        mC = gwr.c();
    }

    GWM_LOG_STAGE("Model fitting");
    mBetas = backfitting(betas);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    // Diagnostic
    GWM_LOG_STAGE("Model Diagnostic");
    vec shat = { 
        mHasHatMatrix ? trace(mS0) : 0,
        mHasHatMatrix ? trace(mS0.t() * mS0) : 0
    };
    mDiagnostic = CalcDiagnostic(mX, mY, shat, mRSS0);
    if (mHasHatMatrix)
    {
        mBetasTV = mBetas / mBetasSE;
    }
    vec yhat = Fitted(mX, mBetas);
    vec residual = mY - yhat;

    // Cleaning
#ifdef ENABLE_CUDA
    if (mParallelType & ParallelType::CUDA)
    {
        cublasDestroy(cubase::handle);
    }
#endif // ENABLE_CUDA

    return mBetas;
}

void GWRMultiscale::createInitialDistanceParameter()
{//回归距离计算
    if (mInitSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mInitSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mInitSpatialWeight.distance()->makeParameter({ mCoords, mCoords });
    }
}

mat GWRMultiscale::backfitting(const mat& betas0)
{
    GWM_LOG_MGWR_BACKFITTING("Model fitting with inital bandwidth");
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas = betas0;
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));

    mat idm = eye(nVar, nVar);
    if (mHasHatMatrix)
    {
        for (uword i = 0; i < nVar; ++i)
        {
            for (uword j = 0; j < nDp ; ++j)
            {
                mSArray.slice(i).row(j) = mX(j, i) * (idm.row(i) * mC.slice(j));
            }
        }
    }

    // ***********************************************************
    // Select the optimum bandwidths for each independent variable
    // ***********************************************************
    GWM_LOG_MGWR_BACKFITTING("Selecting the optimum bandwidths for each independent variable");
    uvec bwChangeNo(nVar, fill::zeros);
    vec resid = mY - Fitted(mX, betas);
    double RSS0 = sum(resid % resid), RSS1 = DBL_MAX;
    double criterion = DBL_MAX;
    for (size_t iteration = 1; iteration <= mMaxIteration && criterion > mCriterionThreshold; iteration++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        GWM_LOG_MGWR_BACKFITTING("#iteration " + to_string(iteration));
        for (uword i = 0; i < nVar  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            mXi = mX.col(i);
            vec fi = betas.col(i) % mX.col(i);
            mYi = resid + fi;
            if (mBandwidthInitilize[i] != BandwidthInitilizeType::Specified)
            {
                GWM_LOG_MGWR_BACKFITTING("#variable-bandwidth-selection " + to_string(i));
                mBandwidthSizeCriterion = bandwidthSizeCriterionVar(mBandwidthSelectionApproach[i]);
                mBandwidthSelectionCurrentIndex = i;
                BandwidthWeight* bwi0 = bandwidth(i);
                bool adaptive = bwi0->adaptive();
                BandwidthSelector selector;
                selector.setBandwidth(bwi0);
                double maxDist = mMaxDistances[i];
                selector.setLower(mGoldenLowerBounds.value_or(adaptive ? mAdaptiveLower : maxDist / 5000.0));
                selector.setUpper(mGoldenUpperBounds.value_or(adaptive ? mCoords.n_rows : maxDist));
                BandwidthWeight* bwi = selector.optimize(this);
                double bwi0s = bwi0->bandwidth(), bwi1s = bwi->bandwidth();
                vector<string> vbs_args {
                    to_string(i),
                    to_string(bwi0s),
                    to_string(bwi1s),
                    to_string(abs(bwi1s - bwi0s))
                };
                if (abs(bwi1s - bwi0s) > mBandwidthSelectThreshold[i])
                {
                    bwChangeNo(i) = 0;
                    vbs_args.push_back("false");
                }
                else
                {
                    bwChangeNo(i) += 1;
                    if (bwChangeNo(i) >= mBandwidthSelectRetryTimes)
                    {
                        mBandwidthInitilize[i] = BandwidthInitilizeType::Specified;
                        vbs_args.push_back("true");
                    }
                    else
                    {
                        vbs_args.push_back(to_string(bwChangeNo(i)));
                        vbs_args.push_back(to_string(mBandwidthSelectRetryTimes - bwChangeNo(i)));
                    }
                }
                mSpatialWeights[i].setWeight(bwi);
#ifdef ENABLE_CUDA
                mSpatialWeights[i].prepareCuda(mGpuId);
#endif // ENABLE_CUDA

                GWM_LOG_MGWR_BACKFITTING("#variable-bandwidth-selection " + strjoin(",", vbs_args));
            }
            GWM_LOG_STOP_BREAK(mStatus);

            betas.col(i) = (this->*mFitVar)(i);
            resid = mY - Fitted(mX, betas);
        }
        RSS1 = RSS(mX, mY, betas);
        criterion = (mCriterionType == BackFittingCriterionType::CVR) ?
                    abs(RSS1 - RSS0) :
                    sqrt(abs(RSS1 - RSS0) / RSS1);
        GWM_LOG_MGWR_BACKFITTING("#backfitting-criterion " + to_string(criterion));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(criterion - mCriterionThreshold)));
        RSS0 = RSS1;
    }
    GWM_LOG_STOP_RETURN(mStatus, betas);

    GWM_LOG_MGWR_BACKFITTING("Finished");
    mRSS0 = RSS0;
    return betas;
}

arma::vec gwm::GWRMultiscale::fitVarBase(const size_t var)
{
    mat S;
    vec beta = (this->*mFitVarCore)(mXi, mYi, mSpatialWeights[var], S);
    if (mHasHatMatrix)
    {
        mat SArrayi = mSArray.slice(var) - mS0;
        mSArray.slice(var) = S * SArrayi + S;
        mS0 = mSArray.slice(var) - SArrayi;
    }
    return beta;
}

double gwm::GWRMultiscale::bandwidthSizeCriterionVarCVBase(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeights[mBandwidthSelectionCurrentIndex].distance());
    try
    {
        vec betas = (this->*mFitVarCoreCV)(mXi, mYi, sw);
        vec res = mYi - mXi % betas;
        double cv = sum(res % res);
        if (mStatus == Status::Success && isfinite(cv))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv));
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
            mBandwidthLastCriterion = cv;
            return cv;
        }
        else return DBL_MAX;
    }
    catch(const std::exception& e)
    {
        return DBL_MAX;
    }
}

double gwm::GWRMultiscale::bandwidthSizeCriterionVarAICBase(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeights[mBandwidthSelectionCurrentIndex].distance());
    try
    {
        vec shat;
        mat betas = (this->*mFitVarCoreSHat)(mXi, mYi, sw, shat);
        double value = GWRBase::AICc(mXi, mYi, betas, shat);
        if (mStatus == Status::Success && isfinite(value))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value));
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else return DBL_MAX;
    }
    catch(const std::exception& e)
    {
        return DBL_MAX;
    }
}

bool GWRMultiscale::isValid()
{
    if (!(mX.n_cols > 0))
        return false;

    size_t nVar = mX.n_cols;

    if (mSpatialWeights.size() != nVar)
        return false;

    if (mBandwidthInitilize.size() != nVar)
        return false;

    if (mBandwidthSelectionApproach.size() != nVar)
        return false;

    if (mPreditorCentered.size() != nVar)
        return false;

    if (mBandwidthSelectThreshold.size() != nVar)
        return false;

    for (size_t i = 0; i < nVar; i++)
    {
        BandwidthWeight* bw = mSpatialWeights[i].weight<BandwidthWeight>();
        if (mBandwidthInitilize[i] == GWRMultiscale::Specified || mBandwidthInitilize[i] == GWRMultiscale::Initial)
        {
            if (bw->adaptive())
            {
                if (bw->bandwidth() <= 1)
                    return false;
            }
            else
            {
                if (bw->bandwidth() < 0.0)
                    return false;
            }
        }
    }

    return true;
}

vec GWRMultiscale::fitVarCoreSerial(const vec &x, const vec &y, const SpatialWeight& sw, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix)
    {
        mat ci, si;
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
        for (uword i = 0; i < nDp  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                ci = xtwx_inv * xtw;
                si = x(i) * ci;
                S.row(mHasHatMatrix ? i : 0) = si;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    else
    {
        for (int i = 0; (uword)i < nDp  ; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

vec GWRMultiscale::fitVarCoreCVSerial(const vec &x, const vec &y, const SpatialWeight& sw)
{
    uword nDp = x.n_rows;
    vec beta(nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = sw.weightVector(i);
        w(i) = 0.0;
        mat xtw = trans(x % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            beta(i) = as_scalar(xtwx_inv * xtwy);
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    return beta;
}

vec GWRMultiscale::fitVarCoreSHatSerial(const vec &x, const vec &y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = x.n_rows;
    vec betas(nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    for (uword i = 0; i < nDp ; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = sw.weightVector(i);
        mat xtw = trans(x % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas(i) = as_scalar(xtwx_inv * xtwy);
            mat ci = xtwx_inv * xtw;
            mat si = x(i) * ci;
            shat(0) += si(0, i);
            shat(1) += as_scalar(si * si.t());
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    return betas;
}


#ifdef ENABLE_OPENMP
vec GWRMultiscale::fitVarCoreOmp(const vec &x, const vec &y, const SpatialWeight& sw, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    bool success = true;
    std::exception except;
    if (mHasHatMatrix)
    {
        S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; (uword)i < nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = x(i) * ci;
                S.row(mHasHatMatrix ? i : 0) = si;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    else
    {
#pragma omp parallel for num_threads(mOmpThreadNum)
        for (int i = 0; (uword)i < nDp; i++)
        {
            GWM_LOG_STOP_CONTINUE(mStatus);
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
            GWM_LOG_PROGRESS(i + 1, nDp);
        }
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

vec GWRMultiscale::fitVarCoreCVOmp(const vec &x, const vec &y, const SpatialWeight& sw)
{
    uword nDp = mCoords.n_rows;
    vec beta(nDp, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            vec w = sw.weightVector(i);
            w(i) = 0.0;
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                beta(i) = as_scalar(xtwx_inv * xtwy);
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    return beta;
}

vec GWRMultiscale::fitVarCoreSHatOmp(const vec &x, const vec &y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = mCoords.n_rows;
    vec betas(nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec w = sw.weightVector(i);
            mat xtw = trans(x % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas(i) = as_scalar(xtwx_inv * xtwy);
                mat ci = xtwx_inv * xtw;
                mat si = x(i) * ci;
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    shat = sum(shat_all, 1);
    return betas;
}

#endif

#ifdef ENABLE_CUDA
vec GWRMultiscale::fitVarCoreCuda(const vec &x, const vec &y, const SpatialWeight& sw, mat &S)
{
    uword nDp = mCoords.n_rows;
    mat betas(1, nDp, fill::zeros);
    mat xt = trans(x);
    cumat u_xt(xt), u_y(y);
    cumat u_betas(1, nDp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(1, nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    size_t groups = nDp / mGroupLength + (nDp % mGroupLength == 0 ? 0 : 1);
    S = mat(mHasHatMatrix ? nDp : 1, nDp, fill::zeros);
    if (mHasHatMatrix)
    {
        mat sg(nDp, mGroupLength, fill::zeros);
        for (size_t i = 0; i < groups; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            size_t begin = i * mGroupLength, length = (begin + mGroupLength > nDp) ? (nDp - begin) : mGroupLength;
            for (size_t j = 0, e = begin + j; j < length; j++, e++)
            {
                checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
                u_xtw.strides(j) = u_xt.diagmul(u_weights);
            }
            custride u_xtwx = u_xtw * u_xt.t();
            custride u_xtwy = u_xtw * u_y;
            custride u_xtwxI = u_xtwx.inv(d_info);
            checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < mGroupLength; j++)
            {
                if (p_info[j] != 0)
                {
                    std::runtime_error e("Cuda failed to get the inverse of matrix");
                    GWM_LOG_ERROR(e.what());
                    throw e;
                }
            }
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
            custride u_c = u_xtwxI * u_xtw;
            custride u_s = u_xt.as_stride().strides(begin, begin + length).t() * u_c;
            u_s.get(sg.memptr());
            S.rows(begin, begin + length - 1) = sg.head_cols(length).t();
            GWM_LOG_PROGRESS(begin + length, nDp);
        }
    }
    else
    {
        for (size_t i = 0; i < groups; i++)
        {
            GWM_LOG_STOP_BREAK(mStatus);
            size_t begin = i * mGroupLength, length = (begin + mGroupLength > nDp) ? (nDp - begin) : mGroupLength;
            for (size_t j = 0, e = begin + j; j < length; j++, e++)
            {
                checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
                u_xtw.strides(j) = u_xt.diagmul(u_weights);
            }
            custride u_xtwx = u_xtw * u_xt.t();
            custride u_xtwy = u_xtw * u_y;
            custride u_xtwxI = u_xtwx.inv(d_info);
            checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
            for (size_t j = 0; j < mGroupLength; j++)
            {
                if (p_info[j] != 0)
                {
                    std::runtime_error e("Cuda failed to get the inverse of matrix");
                    GWM_LOG_ERROR(e.what());
                    throw e;
                }
            }
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
            GWM_LOG_PROGRESS(begin + length, nDp);
        }
        
    }
    u_betas.get(betas.memptr());
    return betas.t();
}

vec GWRMultiscale::fitVarCoreCVCuda(const vec &x, const vec &y, const SpatialWeight& sw)
{
    uword nDp = mCoords.n_rows;
    size_t elems = nDp;
    constexpr size_t nVar = 1;
    cumat u_xt(mXi.t()), u_y(mYi);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    cumat u_betas(1, nDp);
    custride u_xtw(nVar, nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    bool success = true;
    size_t groups = nDp / mGroupLength + (nDp % mGroupLength == 0 ? 0 : 1);
    for (size_t i = 0; i < groups && success; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        size_t begin = i * mGroupLength, length = (begin + mGroupLength > nDp) ? (nDp - begin) : mGroupLength;
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
            checkCudaErrors(cudaMemcpy(u_weights.dmem() + e, &cubase::beta0, sizeof(double), cudaMemcpyHostToDevice));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        custride u_xtwx = u_xtw * u_xt.t();
        custride u_xtwy = u_xtw * u_y;
        custride u_xtwxI = u_xtwx.inv(d_info);
        checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < mGroupLength; j++)
        {
            if (p_info[j] != 0)
            {
                GWM_LOG_ERROR("Cuda failed to get the inverse of matrix");
                success = false;
                break;
            }
        }
        if (success)
        {
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
        }
    }
    checkCudaErrors(cudaFree(d_info));
    delete[] p_info;
    vec betas(nDp, fill::zeros);
    u_betas.get(betas.memptr());
    return betas;
}

vec GWRMultiscale::fitVarCoreSHatCuda(const vec &x, const vec &y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = mCoords.n_rows;
    size_t elems = nDp;
    constexpr size_t nVar = 1;
    cumat u_xt(mXi.t()), u_y(mYi), u_betas(nVar, nDp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(nVar, nDp, mGroupLength);
    mat betas(nVar, nDp);
    shat = vec(2);
    vec sst(mGroupLength);
    mat sg(nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    bool success = true;
    size_t groups = nDp / mGroupLength + (nDp % mGroupLength == 0 ? 0 : 1);
    for (size_t i = 0; i < groups && success; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        size_t begin = i * mGroupLength, length = (begin + mGroupLength > nDp) ? (nDp - begin) : mGroupLength;
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            checkCudaErrors(sw.weightVector(e, u_dists.dmem(), u_weights.dmem()));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        custride u_xtwx = u_xtw * u_xt.t();
        custride u_xtwy = u_xtw * u_y;
        custride u_xtwxI = u_xtwx.inv(d_info);
        checkCudaErrors(cudaMemcpy(p_info, d_info, sizeof(int) * mGroupLength, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < mGroupLength; j++)
        {
            if (p_info[j] != 0)
            {
                GWM_LOG_ERROR("Cuda failed to get the inverse of matrix");
                success = false;
                break;
            }
        }
        if (success)
        {
            u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
            custride u_c = u_xtwxI * u_xtw;
            custride u_s = u_xt.as_stride().strides(begin, begin + length).t() * u_c;
            custride u_sst = u_s * u_s.t();
            u_s.get(sg.memptr());
            u_sst.get(sst.memptr());
            shat(0) += trace(sg.submat(begin, 0, arma::SizeMat(length, length)));
            shat(1) += sum(sst);
        }
    }
    checkCudaErrors(cudaFree(d_info));
    delete[] p_info;
    u_betas.get(betas.memptr());
    return betas.t();
}

#endif // ENABLE_CUDA

GWRMultiscale::BandwidthSizeCriterionFunction GWRMultiscale::bandwidthSizeCriterionVar(GWRMultiscale::BandwidthSelectionCriterionType type)
{
    // if (mParallelType & ParallelType::MPI)
    // {
    switch (type)
    {
    case BandwidthSelectionCriterionType::AIC:
        return &GWRMultiscale::bandwidthSizeCriterionVarAICBase;
    default:
        return &GWRMultiscale::bandwidthSizeCriterionVarCVBase;
    }
    // }
}

void GWRMultiscale::setParallelType(const ParallelType &type)
{
    if (parallelAbility() & type)
    {
        mParallelType = type;
        switch (type) {
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mFitVarCore = &GWRMultiscale::fitVarCoreOmp;
            mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVOmp;
            mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatOmp;
            break;
#endif
#ifdef ENABLE_CUDA
        case ParallelType::CUDA:
            mFitVarCore = &GWRMultiscale::fitVarCoreCuda;
            mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVCuda;
            mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatCuda;
            break;
#endif
        default:
            mFitVarCore = &GWRMultiscale::fitVarCoreSerial;
            mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVSerial;
            mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatSerial;
            break;
        }
    }
}

void GWRMultiscale::setSpatialWeights(const vector<SpatialWeight> &spatialWeights)
{
    SpatialMultiscaleAlgorithm::setSpatialWeights(spatialWeights);
    if (spatialWeights.size() > 0)
    {
        setInitSpatialWeight(spatialWeights[0]);
    }
}

void GWRMultiscale::setBandwidthSelectionApproach(const vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach)
{
    if (bandwidthSelectionApproach.size() == mX.n_cols)
    {
        mBandwidthSelectionApproach = bandwidthSelectionApproach;
    }
    else
    {
        length_error e("bandwidthSelectionApproach size do not match indepvars");
        GWM_LOG_ERROR(e.what());
        throw e;
    }  
}

void GWRMultiscale::setBandwidthInitilize(const vector<BandwidthInitilizeType> &bandwidthInitilize)
{
    if(bandwidthInitilize.size() == mX.n_cols){
        mBandwidthInitilize = bandwidthInitilize;
    }
    else
    {
        length_error e("BandwidthInitilize size do not match indepvars");
        GWM_LOG_ERROR(e.what());
        throw e;
    }   
}
