#include "GWRBasic.h"
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include "Logger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "CudaUtils.h"
#include "cumat.hpp"
#endif

#include "mpi.h"

using namespace std;
using namespace arma;
using namespace gwm;

enum class GWRBasicFitMpiTags
{
    Betas = 1 << 0,
    BetasSE,
    SHat,
    QDiag,
    SMat,
    Finished
};

enum class GWRBasicAICMpiTags
{
    SHat = 1 << 4,
    Betas
};

enum class GWRBasicCVMpiTags
{
    Betas = 1 << 8
};

RegressionDiagnostic GWRBasic::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    double n = (double)x.n_rows;
    double AIC = n * log(rss / n) + n * log(2 * datum::pi) + n + shat(0);
    double AICc = n * log(rss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    double edf = n - 2 * shat(0) + shat(1);
    double enp = 2 * shat(0) - shat(1);
    double yss = sum((y - mean(y)) % (y - mean(y)));
    double r2 = 1 - rss / yss;
    double r2_adj = 1 - (1 - r2) * (n - 1) / (edf - 1);
    return { rss, AIC, AICc, enp, edf, r2, r2_adj };
}

mat GWRBasic::fit()
{
    GWM_LOG_STAGE("Initializing");
    uword nDp = mCoords.n_rows, nVars = mX.n_cols;
    if (mParallelType & ParallelType::MPI)
    {
        uword aShape[4];
        if (mWorkerId == 0) // master process
        {
            // sync matrix dimensions
            uword nDim = mCoords.n_cols;
            mWorkRangeSize = nDp / uword(mWorkerNum) + ((nDp % uword(mWorkerNum) == 0) ? 0 : 1);
            aShape[0] = nDp;
            aShape[1] = nVars;
            aShape[2] = nDim;
            aShape[3] = mWorkRangeSize;
            // cout << mWorkerId << " process send size: [" << aShape[0] << "," << aShape[1] << "," << aShape[2] << "," << aShape[3] << "]\n";
        }
        MPI_Bcast(aShape, 4, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        if (mWorkerId > 0)                // slave process
        {
            // cout << mWorkerId << " process recv size: [" << aShape[0] << "," << aShape[1] << "," << aShape[2] << "," << aShape[3] << "]\n";
            nDp = aShape[0];
            nVars = aShape[1];
            mX.resize(nDp, nVars);
            mY.resize(nDp);
            mCoords.resize(aShape[0], aShape[2]);
            mWorkRangeSize = aShape[3];
        }
        MPI_Bcast(mX.memptr(), mX.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(mY.memptr(), mY.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(mCoords.memptr(), mCoords.n_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        uword workRangeFrom = uword(mWorkerId) * mWorkRangeSize;
        // cout << mWorkerId << " process work range: [" << workRangeFrom << "," << min(workRangeFrom + mWorkRangeSize, nDp) << "]\n";
        mWorkRange = make_pair(workRangeFrom, min(workRangeFrom + mWorkRangeSize, nDp));
        // cout << mWorkerId << " process work range: [" << mWorkRange.value().first << "," << mWorkRange.value().second << "]\n";
    }
    createDistanceParameter();
#ifdef ENABLE_CUDA
    if (mParallelType == ParallelType::CUDA)
    {
        cublasCreate(&cubase::handle);
        mSpatialWeight.prepareCuda(mGpuId);
    }
#endif // ENABLE_CUDA
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

    if (mIsAutoselectIndepVars)
    {
        GWM_LOG_STAGE("Independent variable selection");
        vector<size_t> indep_vars;
        for (size_t i = (mHasIntercept ? 1 : 0); i < mX.n_cols; i++)
        {
            indep_vars.push_back(i);
        }
        size_t k = indep_vars.size();
        mIndepVarSelectionProgressTotal = (k + 1) * k / 2;
        mIndepVarSelectionProgressCurrent = 0;
        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion());
        VariableForwardSelector selector(indep_vars, mIndepVarSelectionThreshold);
        mSelectedIndepVars = selector.optimize(this);
        if (mSelectedIndepVars.size() > 0)
        {
            mX = mX.cols(VariableForwardSelector::index2uvec(mSelectedIndepVars, mHasIntercept));
            mIndepVarsSelectionCriterionList = selector.indepVarsCriterion();
        }
        nVars = mX.n_cols;
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    }

    if (mIsAutoselectBandwidth)
    {
        GWM_LOG_STAGE("Bandwidth selection");
        BandwidthWeight* bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = mGoldenLowerBounds.value_or(bw0->adaptive() ? 20 : 0.0);
        double upper = mGoldenUpperBounds.value_or(bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance());

        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bw0));
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
#ifdef ENABLE_CUDA
            if (mParallelType == ParallelType::CUDA)
            {
                mSpatialWeight.prepareCuda(mGpuId);
            }
#endif // ENABLE_CUDA
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
        GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    }

    GWM_LOG_STAGE("Model fitting");
    mBetas = (this->*mFitFunction)();
    
#ifdef ENABLE_CUDA
    if (mParallelType == ParallelType::CUDA)
    {
        cublasDestroy(cubase::handle);
    }
#endif // ENABLE_CUDA

    return mBetas;
}

mat GWRBasic::predict(const mat& locations)
{
    GWM_LOG_STAGE("Initialization");
    uword nDp = mCoords.n_rows, nVars = mX.n_cols;
    createPredictionDistanceParameter(locations);
#ifdef ENABLE_CUDA
    if (mParallelType == ParallelType::CUDA)
    {
        cublasCreate(&cubase::handle);
        mSpatialWeight.prepareCuda(mGpuId);
    }
#endif // ENABLE_CUDA
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));
    
    GWM_LOG_STAGE("Prediction");
    mBetas = (this->*mPredictFunction)(locations, mX, mY);
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVars, arma::fill::zeros));

#ifdef ENABLE_CUDA
    if (mParallelType == ParallelType::CUDA)
    {
        cublasDestroy(cubase::handle);
    }
#endif // ENABLE_CUDA

    return mBetas;
}

void GWRBasic::createPredictionDistanceParameter(const arma::mat& locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == Distance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({ locations, mCoords });
    }
}

mat GWRBasic::predictSerial(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        mat xtw = trans(x.each_col() % w);
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
            throw e;
        }
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    return betas.t();
}

arma::mat gwm::GWRBasic::fitBase()
{
    mBetas = (this->*mFitCoreFunction)(mX, mY, mSpatialWeight, mBetasSE, mSHat, mQDiag, mS);
    mDiagnostic = CalcDiagnostic(mX, mY, mBetas, mSHat);
    double trS = mSHat(0), trStS = mSHat(1);
    double nDp = double(mCoords.n_rows);
    double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
    mBetasSE = sqrt(sigmaHat * mBetasSE);
    return mBetas;
}

mat GWRBasic::fitCoreSerial(const mat &x, const vec &y, const SpatialWeight &sw, mat &betasSE, vec &shat, vec &qDiag, mat &S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    qDiag = vec(nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    std::pair<uword, uword> workRange = mWorkRange.value_or(make_pair(0, nDp));
    // cout << mWorkerId << " process work range: [" << workRange.first << "," << workRange.second << "]\n";
    for (uword i = workRange.first; i < workRange.second; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = mSpatialWeight.weightVector(i);
        mat xtw = trans(x.each_col() % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            betasSE.col(i) = sum(ci % ci, 1);
            mat si = x.row(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
            vec p = - si.t();
            p(i) += 1.0;
            qDiag += p % p;
            S.row(isStoreS() ? i : 0) = si;
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    betasSE = betasSE.t();
    return betas.t();
}

mat GWRBasic::fitCoreCVSerial(const mat& x, const vec& y, const SpatialWeight& sw)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    vec shat(2, fill::zeros);
    mat betas(nVar, nDp, fill::zeros);
    std::pair<uword, uword> workRange = mWorkRange.value_or(make_pair(0, nDp));
    for (uword i = workRange.first; i < workRange.second; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = sw.weightVector(i);
        w(i) = 0.0;
        mat xtw = trans(x.each_col() % w);
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
            throw e;
        }
    }
    return betas.t();
}

mat GWRBasic::fitCoreSHatSerial(const mat& x, const vec& y, const SpatialWeight& sw, vec& shat)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    std::pair<uword, uword> workRange = mWorkRange.value_or(make_pair(0, nDp));
    for (uword i = workRange.first; i < workRange.second; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        vec w = sw.weightVector(i);
        mat xtw = trans(x.each_col() % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
            mat ci = xtwx_inv * xtw;
            mat si = x.row(i) * ci;
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            throw e;
        }
    }
    return betas.t();
}

double GWRBasic::bandwidthSizeCriterionCV(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeight.distance());
    try
    {
        mat betas = (this->*mFitCoreCVFunction)(mX, mY, sw);
        vec res = mY - sum(mX % betas, 1);
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

double GWRBasic::bandwidthSizeCriterionAIC(BandwidthWeight* bandwidthWeight)
{
    SpatialWeight sw(bandwidthWeight, mSpatialWeight.distance());
    try
    {
        vec shat(2, fill::zeros);
        mat betas = (this->*mFitCoreSHatFunction)(mX, mY, sw, shat);
        double value = GWRBase::AICc(mX, mY, betas, shat);
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

double gwm::GWRBasic::indepVarsSelectionCriterion(const vector<size_t>& indepVars)
{
    mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    vec y = mY;
    try
    {
        vec shat(2, arma::fill::zeros);
        mat betas = (this->*mFitCoreSHatFunction)(x, y, mSpatialWeight, shat);
        GWM_LOG_PROGRESS(++mIndepVarSelectionProgressCurrent, mIndepVarSelectionProgressTotal);
        if (mStatus == Status::Success)
        {
            double aic = GWRBase::AICc(x, y, betas, shat);
            GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion(indepVars, aic));
            return aic;
        }
        else return DBL_MAX;
    }
    catch(const std::exception& e)
    {
        return DBL_MAX;
    }
}

#ifdef ENABLE_OPENMP
mat GWRBasic::predictOmp(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nRp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (success)
        {
            vec w = mSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
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
        }
        GWM_LOG_PROGRESS(i + 1, nRp);
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

mat GWRBasic::fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    mat qDiag_all(nDp, mOmpThreadNum, fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (success)
        {
            int thread = omp_get_thread_num();
            vec w = mSpatialWeight.weightVector(i);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                betasSE.col(i) = sum(ci % ci, 1);
                mat si = x.row(i) * ci;
                shat_all(0, thread) += si(0, i);
                shat_all(1, thread) += det(si * si.t());
                vec p = - si.t();
                p(i) += 1.0;
                qDiag_all.col(thread) += p % p;
                S.row(isStoreS() ? i : 0) = si;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                except = e;
                success = false;
            }
        }
        GWM_LOG_PROGRESS(i + 1, nDp);
    }
    if (!success)
    {
        throw except;
    }
    shat = sum(shat_all, 1);
    qDiag = sum(qDiag_all, 1);
    betasSE = betasSE.t();
    return betas.t();
}

double GWRBasic::bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            w(i) = 0.0;
            mat xtw = trans(mX.each_col() % w);
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                vec beta = xtwx_inv * xtwy;
                double res = mY(i) - det(mX.row(i) * beta);
                if (isfinite(res))
                    cv_all(thread) += res * res;
                else
                    flag = false;
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    if (mStatus == Status::Success && flag)
    {
        double cv = sum(cv_all);
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

double GWRBasic::bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeight.distance()->distance(i);
            vec w = bandwidthWeight->weight(d);
            mat xtw = trans(mX.each_col() % w);
            mat xtwx = xtw * mX;
            mat xtwy = xtw * mY;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = mX.row(i) * ci;
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
    if (mStatus == Status::Success && flag)
    {
        vec shat = sum(shat_all, 1);
        double value = GWRBase::AICc(mX, mY, betas.t(), shat);
        if (isfinite(value))
        {
            GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, value));
            GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - value)));
            mBandwidthLastCriterion = value;
            return value;
        }
        else return DBL_MAX;
    }
    else return DBL_MAX;
}

double GWRBasic::indepVarsSelectionCriterionOmp(const vector<size_t>& indepVars)
{
    mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    vec y = mY;
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mat shat(2, mOmpThreadNum, fill::zeros);
    int flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        GWM_LOG_STOP_CONTINUE(mStatus);
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec w(nDp, fill::ones);
            mat xtw = trans(x.each_col() % w);
            mat xtwx = xtw * x;
            mat xtwy = xtw * y;
            try
            {
                mat xtwx_inv = inv_sympd(xtwx);
                betas.col(i) = xtwx_inv * xtwy;
                mat ci = xtwx_inv * xtw;
                mat si = x.row(i) * ci;
                shat(0, thread) += si(0, i);
                shat(1, thread) += det(si * si.t());
            }
            catch (const exception& e)
            {
                GWM_LOG_ERROR(e.what());
                flag = false;
            }
        }
    }
    GWM_LOG_PROGRESS(++mIndepVarSelectionProgressCurrent, mIndepVarSelectionProgressTotal);
    if (mStatus == Status::Success && flag)
    {
        double value = GWRBase::AICc(x, y, betas.t(), sum(shat, 1));
        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion(indepVars, value));
        return value;
    }
    else return DBL_MAX;
}
#endif

#ifdef ENABLE_CUDA

mat GWRBasic::fitCuda(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat xt = trans(x);
    mat betas = mat(nVar, nDp, arma::fill::zeros);
    betasSE = mat(nVar, nDp, arma::fill::zeros);
    shat = vec(2, arma::fill::zeros);
    qDiag = vec(nDp, arma::fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, arma::fill::zeros);
    size_t groups = nDp / mGroupLength + (nDp % mGroupLength == 0 ? 0 : 1);
    cumat u_xt(xt), u_y(y);
    cumat u_betas(nVar, nDp), u_betasSE(nVar, nDp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(nVar, nDp, mGroupLength);//, u_s(1, nDp, mGroupLength);
    custride u_xtwx(nVar, nVar, mGroupLength), u_xtwy(nVar, 1, mGroupLength), u_xtwxI(nVar, nVar, mGroupLength);
    custride u_c(nVar, nDp, mGroupLength), u_s(1, nDp, mGroupLength);
    custride u_cct(nVar, nVar, mGroupLength), u_sst(1, 1, mGroupLength);
    mat sg(nDp, mGroupLength, fill::zeros);
    vec sst(mGroupLength, fill::zeros);
    cube cct(nVar, nVar, mGroupLength, fill::zeros);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    for (size_t i = 0; i < groups; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        size_t begin = i * mGroupLength, length = (begin + mGroupLength > nDp) ? (nDp - begin) : mGroupLength;
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            checkCudaErrors(mSpatialWeight.weightVector(e, u_dists.dmem(), u_weights.dmem()));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        // xtwx and xtwy
        // xtw * x [k*n,n*k]
        u_xtwx = u_xtw * u_xt.t();
        // xtw * y [k*n,n*1]
        u_xtwy = u_xtw * u_y;
        // inv
        u_xtwxI = u_xtwx.inv(d_info);
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
        // beta = xtwxI * xtwy [k*k,k*1]
        u_betas.as_stride().strides(begin, begin + length) = u_xtwxI * u_xtwy;
        // ci = xtwxI * xtw [k*k,t(n*k)]
        u_c = u_xtwxI * u_xtw;
        // si = t(xti) * ci [1*k,k*n]
        u_s = u_xt.as_stride().strides(begin, begin + length).t() * u_c;
        u_s.get(sg.memptr());
        shat(0) += trace(sg.submat(begin, 0, arma::SizeMat(length, length)));
        // cct = ci * cit [k*n,t(k*n)]
        u_cct = u_c * u_c.t();
        u_sst = u_s * u_s.t();
        u_cct.get(cct.memptr());
        u_sst.get(sst.memptr());
        shat(1) += sum(sst.head(length));
        // Transfer to cpu Perform further diagnostic
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            betasSE.col(e) = diagvec(cct.slice(j));
            vec p = -sg.col(j);
            p(e) += 1.0;
            qDiag += p % p;
            S.row(isStoreS() ? e : 0) = sg.col(j).t();
        }
        GWM_LOG_PROGRESS(begin + length, nDp);
    }
    u_betas.get(betas.memptr());
    betasSE = betasSE.t();
    cudaFree(d_info);
    return betas.t();
}

arma::mat GWRBasic::predictCuda(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, fill::zeros);
    size_t groups = nRp / mGroupLength + (nRp % mGroupLength == 0 ? 0 : 1);
    cumat u_xt(x.t()), u_y(y);
    custride u_xtwx(nVar, nVar, mGroupLength), u_xtwy(nVar, 1, mGroupLength), u_xtwxI(nVar, nVar, mGroupLength);
    cumat u_betas(nVar, nRp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(nVar, nDp, mGroupLength);
    int *d_info, *p_info;
    p_info = new int[mGroupLength];
    checkCudaErrors(cudaMalloc(&d_info, sizeof(int) * mGroupLength));
    for (size_t i = 0; i < groups; i++)
    {
        GWM_LOG_STOP_BREAK(mStatus);
        size_t begin = i * mGroupLength, length = (begin + mGroupLength > nRp) ? (nRp - begin) : mGroupLength;
        for (size_t j = 0, e = begin + j; j < length; j++, e++)
        {
            checkCudaErrors(mSpatialWeight.weightVector(e, u_dists.dmem(), u_weights.dmem()));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        u_xtwx = u_xtw * u_xt.t();
        u_xtwy = u_xtw * u_y;
        u_xtwxI = u_xtwx.inv(d_info);
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
    cudaFree(d_info);
    delete[] p_info;
    u_betas.get(betas.memptr());
    return betas.t();
}

double GWRBasic::bandwidthSizeCriterionCVCuda(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    size_t elems = nDp;
    cumat u_xt(mX.t()), u_y(mY);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(nVar, nDp, mGroupLength);
    custride u_xtwx(nVar, nVar, mGroupLength), u_xtwy(nVar, 1, mGroupLength), u_xtwxI(nVar, nVar, mGroupLength);
    custride u_betas(nVar, 1, mGroupLength), u_yhat(1, 1, mGroupLength);
    vec yhat(mGroupLength), yhat_all(nDp);
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
            checkCudaErrors(mSpatialWeight.distance()->distance(e, u_dists.dmem(), &elems));
            checkCudaErrors(bandwidthWeight->weight(u_dists.dmem(), u_weights.dmem(), elems));
            checkCudaErrors(cudaMemcpy(u_weights.dmem() + e, &cubase::beta0, sizeof(double), cudaMemcpyHostToDevice));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        u_xtwx = u_xtw * u_xt.t();
        u_xtwy = u_xtw * u_y;
        u_xtwxI = u_xtwx.inv(d_info);
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
            u_betas = u_xtwxI * u_xtwy;
            u_yhat = u_xt.as_stride().strides(begin, begin + length).t() * u_betas;
            u_yhat.get(yhat.memptr());
            yhat_all.rows(begin, begin + length - 1) = yhat.head_rows(length);
        }
    }
    checkCudaErrors(cudaFree(d_info));
    delete[] p_info;
    if (!success) return DBL_MAX;
    double cv = as_scalar((mY - yhat_all).t() * (mY - yhat_all));
    if (isfinite(cv))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, cv));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - cv)));
        mBandwidthLastCriterion = cv;
        return cv;
    }
    else return DBL_MAX;
}

double GWRBasic::bandwidthSizeCriterionAICCuda(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    size_t elems = nDp;
    cumat u_xt(mX.t()), u_y(mY), u_betas(nVar, nDp);
    cumat u_dists(nDp, 1), u_weights(nDp, 1);
    custride u_xtw(nVar, nDp, mGroupLength);
    custride u_xtwx(nVar, nVar, mGroupLength), u_xtwy(nVar, 1, mGroupLength), u_xtwxI(nVar, nVar, mGroupLength);
    custride u_c(nVar, nDp, mGroupLength), u_s(1, nDp, mGroupLength);
    custride u_sst(1, 1, mGroupLength);
    mat betas(nVar, nDp);
    vec shat(2), sst(mGroupLength);
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
            checkCudaErrors(mSpatialWeight.distance()->distance(e, u_dists.dmem(), &elems));
            checkCudaErrors(bandwidthWeight->weight(u_dists.dmem(), u_weights.dmem(), elems));
            u_xtw.strides(j) = u_xt.diagmul(u_weights);
        }
        u_xtwx = u_xtw * u_xt.t();
        u_xtwy = u_xtw * u_y;
        u_xtwxI = u_xtwx.inv(d_info);
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
            u_c = u_xtwxI * u_xtw;
            u_s = u_xt.as_stride().strides(begin, begin + length).t() * u_c;
            u_s.get(sg.memptr());
            shat(0) += trace(sg.submat(begin, 0, arma::SizeMat(length, length)));
            u_sst = u_s * u_s.t();
            u_sst.get(sst.memptr());
            shat(1) += sum(sst);
        }
    }
    checkCudaErrors(cudaFree(d_info));
    delete[] p_info;
    if (!success) return DBL_MAX;
    u_betas.get(betas.memptr());
    double aic = GWRBasic::AICc(mX, mY, betas.t(), shat);
    if (isfinite(aic))
    {
        GWM_LOG_INFO(IBandwidthSelectable::infoBandwidthCriterion(bandwidthWeight, aic));
        GWM_LOG_PROGRESS_PERCENT(exp(- abs(mBandwidthLastCriterion - aic)));
        mBandwidthLastCriterion = aic;
        return aic;
    }
    else return DBL_MAX;
}

double GWRBasic::indepVarsSelectionCriterionCuda(const std::vector<size_t>& indepVars)
{
    mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    mat y = mY;
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    cumat u_xt(x.t()), u_y(y), u_betas(nVar, nDp);
    cumat u_weights(vec(nDp, fill::ones));
    custride u_xtw(nVar, nDp, mGroupLength);
    mat betas(nVar, nDp);
    vec shat(2), sst(mGroupLength);
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
    if (!success) return DBL_MAX;
    u_betas.get(betas.memptr());
    GWM_LOG_PROGRESS(++mIndepVarSelectionProgressCurrent, mIndepVarSelectionProgressTotal);
    double aic = GWRBasic::AICc(x, y, betas.t(), shat);
    if (isfinite(aic))
    {
        GWM_LOG_INFO(IVarialbeSelectable::infoVariableCriterion(indepVars, aic));
        mBandwidthLastCriterion = aic;
        return aic;
    }
    else return DBL_MAX;
}

#endif

#ifdef ENABLE_MPI
double GWRBasic::indepVarsSelectionCriterionMpi(const vector<size_t>& indepVars)
{
    mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
    mat y = mY;
    vec shat(2, arma::fill::zeros);
    double aic;
    mat betas = (this->*mFitCoreSHatFunction)(x, y, mSpatialWeight, shat);
GWM_MPI_MASTER_BEGIN
    vec shat_all = shat;
    umat received(mWorkerNum, 2, arma::fill::zeros);
    received.row(0).fill(1);
    double *buf = new double[x.n_elem];
    while (!all(all(received == 1)))
    {
        MPI_Status status;
        MPI_Recv(&buf, x.n_elem, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        received(status.MPI_SOURCE, status.MPI_TAG) = 1;
        switch (GWRBasicAICMpiTags(status.MPI_TAG))
        {
        case GWRBasicAICMpiTags::SHat:
            shat_all(0) += buf[0];
            shat_all(1) += buf[1];
            break;
        case GWRBasicAICMpiTags::Betas:
            betas += mat(buf, betas.n_rows, betas.n_cols);
            break;
        default:
            break;
        }
    }
    delete[] buf;
    aic = GWRBasic::AICc(x, y, betas, shat_all);
GWM_MPI_MASTER_END
GWM_MPI_WORKER_BEGIN
    MPI_Send(shat.memptr(), 2, MPI_DOUBLE, 0, int(GWRBasicAICMpiTags::SHat), MPI_COMM_WORLD);
    MPI_Send(betas.memptr(), betas.n_elem, MPI_DOUBLE, 0, int(GWRBasicAICMpiTags::Betas), MPI_COMM_WORLD);
GWM_MPI_WORKER_END
    MPI_Bcast(&aic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return aic;
}

double GWRBasic::bandwidthSizeCriterionCVMpi(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mX.n_rows;
    SpatialWeight sw(bandwidthWeight, mSpatialWeight.distance());
    double cv;
    mat betas = (this->*mFitCoreCVFunction)(mX, mY, sw).t();
GWM_MPI_MASTER_BEGIN
    uvec received(mWorkerNum, arma::fill::zeros);
    received.row(0).fill(1);
    unique_ptr<double[], default_delete<double[]>> buf(new double[betas.n_elem]);
    while (!all(received))
    {
        MPI_Status status;
        MPI_Recv(buf.get(), betas.n_elem, MPI_DOUBLE, MPI_ANY_SOURCE, int(GWRBasicCVMpiTags::Betas), MPI_COMM_WORLD, &status);
        received(status.MPI_SOURCE) = 1;
        uword rangeFrom = status.MPI_SOURCE * mWorkRangeSize, rangeTo = min(rangeFrom + mWorkRangeSize, nDp), rangeSize = rangeTo - rangeFrom;
        betas.cols(rangeFrom, rangeTo - 1) = mat(buf.get(), betas.n_rows, rangeSize);
    }
    vec residual = mY - mX % betas.t();
    cv = sum(residual % residual);
GWM_MPI_MASTER_END
GWM_MPI_WORKER_BEGIN
    std::pair<uword, uword> workRange = mWorkRange.value_or(make_pair(0, nDp));
    betas = betas.cols(workRange.first, workRange.second - 1);
    MPI_Send(betas.memptr(), betas.n_elem, MPI_DOUBLE, 0, int(GWRBasicCVMpiTags::Betas), MPI_COMM_WORLD);
GWM_MPI_WORKER_END
    MPI_Bcast(&cv, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return cv;
}

double GWRBasic::bandwidthSizeCriterionAICMpi(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mX.n_rows;
    SpatialWeight sw(bandwidthWeight, mSpatialWeight.distance());
    double aic;
    vec shat(2, fill::zeros);
    mat betas = (this->*mFitCoreSHatFunction)(mX, mY, sw, shat).t();
GWM_MPI_MASTER_BEGIN
    umat received(mWorkerNum, 2, arma::fill::zeros);
    received.row(0).fill(1);
    unique_ptr<double[], default_delete<double[]>> buf(new double[betas.n_elem]);
    while (!all(all(received)))
    {
        MPI_Status status;
        MPI_Recv(buf.get(), betas.n_elem, MPI_DOUBLE, MPI_ANY_SOURCE, int(GWRBasicCVMpiTags::Betas), MPI_COMM_WORLD, &status);
        received(status.MPI_SOURCE) = 1;
        uword rangeFrom = status.MPI_SOURCE * mWorkRangeSize, rangeTo = min(rangeFrom + mWorkRangeSize, nDp), rangeSize = rangeTo - rangeFrom;
        switch (GWRBasicAICMpiTags(status.MPI_TAG))
        {
        case GWRBasicAICMpiTags::SHat:
            shat(0) += buf[0];
            shat(1) += buf[1];
            break;
        case GWRBasicAICMpiTags::Betas:
            betas.cols(rangeFrom, rangeTo - 1) = mat(buf.get(), betas.n_rows, rangeSize);
            break;
        default:
            break;
        }
    }
    aic = GWRBasic::AICc(mX, mY, betas, shat);
GWM_MPI_MASTER_END
GWM_MPI_WORKER_BEGIN
    std::pair<uword, uword> workRange = mWorkRange.value_or(make_pair(0, nDp));
    betas = betas.cols(workRange.first, workRange.second - 1);
    MPI_Send(betas.memptr(), betas.n_elem, MPI_DOUBLE, 0, int(GWRBasicCVMpiTags::Betas), MPI_COMM_WORLD);
GWM_MPI_WORKER_END
    MPI_Bcast(&aic, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return aic;
}

arma::mat gwm::GWRBasic::fitMpi()
{
    // fit GWR on each process
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mBetas = (this->*mFitCoreFunction)(mX, mY, mSpatialWeight, mBetasSE, mSHat, mQDiag, mS).t();
    mBetasSE = mBetasSE.t();
    mS = mS.t();
    // gather results to master process
    GWM_MPI_MASTER_BEGIN
    mat shat_all(2, mWorkerNum);
    mat qdiag_all(nDp, mWorkerNum);
    shat_all.col(0) = mSHat;
    qdiag_all.col(0) = mQDiag;
    // prepare to receive data
    umat received(mWorkerNum, (isStoreS() ? 5 : 4), fill::zeros);
    received.row(0).fill(1);
    int bufSize = isStoreS() ? nDp * nDp : nDp * nVar;
    unique_ptr<double[], std::default_delete<double[]>> buf(new double[bufSize]);
    while (!all(all(received)))
    {
        MPI_Status status;
        // printf("0 process received message with %d from %d\n", status.MPI_TAG, status.MPI_SOURCE);
        MPI_Recv(buf.get(), bufSize, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        received(status.MPI_SOURCE, status.MPI_TAG) = 1;
        uword rangeFrom = status.MPI_SOURCE * mWorkRangeSize, rangeTo = min(rangeFrom + mWorkRangeSize, nDp), rangeSize = rangeTo - rangeFrom;
        switch (GWRBasicFitMpiTags(status.MPI_TAG))
        {
        case GWRBasicFitMpiTags::Betas:
            mBetas.cols(rangeFrom, rangeTo - 1) = mat(buf.get(), nVar, rangeSize);
            break;
        case GWRBasicFitMpiTags::BetasSE:
            mBetasSE.cols(rangeFrom, rangeTo - 1) = mat(buf.get(), nVar, rangeSize);
            break;
        case GWRBasicFitMpiTags::SHat:
            shat_all.col(status.MPI_SOURCE) = vec(buf.get(), 2);
            break;
        case GWRBasicFitMpiTags::QDiag:
            qdiag_all.col(status.MPI_SOURCE) = vec(buf.get(), nDp);
            break;
        case GWRBasicFitMpiTags::SMat:
            mS.cols(rangeFrom, rangeTo - 1) = mat(buf.get(), nDp, rangeSize);
            break;
        default:
            break;
        }
    }
    mBetas = mBetas.t();
    mBetasSE = mBetasSE.t();
    mS = mS.t();
    mSHat = sum(shat_all, 1);
    mQDiag = sum(qdiag_all, 1);
    // printf("shat [%lf,%lf]", mSHat(0), mSHat(1));
    // diagnostic in master process
    GWM_LOG_STAGE("Model Diagnostic");
    mDiagnostic = CalcDiagnostic(mX, mY, mBetas, mSHat);
    double trS = mSHat(0), trStS = mSHat(1);
    double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
    mBetasSE = sqrt(sigmaHat * mBetasSE);    
    // vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
    // vec dyhat2 = (mY - yhat) % (mY - yhat);
    // vec localR2 = vec(nDp, fill::zeros);
    // for (uword i = 0; i < nDp; i++)
    // {
    //     vec w = mSpatialWeight.weightVector(i);
    //     double tss = sum(dybar2 % w);
    //     double rss = sum(dyhat2 % w);
    //     localR2(i) = (tss - rss) / tss;
    // }
    GWM_MPI_MASTER_END
    GWM_MPI_WORKER_BEGIN
    std::pair<uword, uword> workRange = mWorkRange.value_or(make_pair(0, nDp));
    // printf("%d process work range: [%lld, %lld]\n", mWorkerId, workRange.first, workRange.second);
    // cout << mWorkerId << " process work range: [" << workRange.first << "," << workRange.second << "]\n";
    mat betas = mBetas.cols(workRange.first, workRange.second - 1);
    mat betasSE = mBetasSE.cols(workRange.first, workRange.second - 1);
    mat S = mS.cols(workRange.first, workRange.second - 1);
    MPI_Send(betas.memptr(), betas.n_elem, MPI_DOUBLE, 0, int(GWRBasicFitMpiTags::Betas), MPI_COMM_WORLD);
    MPI_Send(betasSE.memptr(), betasSE.n_elem, MPI_DOUBLE, 0, int(GWRBasicFitMpiTags::BetasSE), MPI_COMM_WORLD);
    MPI_Send(mSHat.memptr(), mSHat.n_elem, MPI_DOUBLE, 0, int(GWRBasicFitMpiTags::SHat), MPI_COMM_WORLD);
    MPI_Send(mQDiag.memptr(), mQDiag.n_elem, MPI_DOUBLE, 0, int(GWRBasicFitMpiTags::QDiag), MPI_COMM_WORLD);
    if (isStoreS()) MPI_Send(S.memptr(), S.n_elem, MPI_DOUBLE, 0, int(GWRBasicFitMpiTags::SMat), MPI_COMM_WORLD);
    GWM_MPI_WORKER_END
    
    // check cancel status
    GWM_LOG_STOP_RETURN(mStatus, mat(nDp, nVar, arma::fill::zeros));
    return mBetas;
}

// double GWRBasic::indepVarsSelectionCriterion(const mat& x, const vec& y, vec& shat)
// {
//     try
//     {
//         mat S;
//         mat betas = (this->*mFitCoreSHatFunction)(x, y, mSpatialWeight, shat, S);
//         GWM_LOG_PROGRESS(++mIndepVarSelectionProgressCurrent, mIndepVarSelectionProgressTotal);
//         if (mStatus == Status::Success)
//         {
//             double rss = GWRBase::RSS(x, y, betas);
//             return rss;
//         }
//         else return DBL_MAX;
//     }
//     catch(const std::exception& e)
//     {
//         return DBL_MAX;
//     }
    
// }
#endif // ENABLE_MPI

void GWRBasic::setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion)
{
    switch (criterion)
    {
    case BandwidthSelectionCriterionType::CV:
        mBandwidthSelectionCriterionFunction = &GWRBasic::bandwidthSizeCriterionCV;
        break;
    default:
        mBandwidthSelectionCriterionFunction = &GWRBasic::bandwidthSizeCriterionAIC;
        break;
    }
    mBandwidthSelectionCriterion = criterion;
}

void GWRBasic::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &GWRBasic::predictSerial;
            mFitFunction = &GWRBasic::fitBase;
            mFitCoreFunction = &GWRBasic::fitCoreSerial;
            mFitCoreCVFunction = &GWRBasic::fitCoreCVSerial;
            mFitCoreSHatFunction = &GWRBasic::fitCoreSHatSerial;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterion;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mPredictFunction = &GWRBasic::predictOmp;
            mFitFunction = &GWRBasic::fitBase;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionOmp;
            break;
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        case ParallelType::CUDA:
            mPredictFunction = &GWRBasic::predictCuda;
            mFitFunction = &GWRBasic::fitCuda;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionCuda;
            break;
#endif // ENABLE_CUDA
        default:
            mPredictFunction = &GWRBasic::predictSerial;
            mFitFunction = &GWRBasic::fitBase;
            break;
        }
#ifdef ENABLE_MPI
        if (type & ParallelType::MPI)
        {
            mFitFunction = &GWRBasic::fitMpi;
            mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterionMpi;
        }
#endif
        setBandwidthSelectionCriterion(mBandwidthSelectionCriterion);
    }
}

bool GWRBasic::isValid()
{
    if (GWRBase::isValid())
    {
        double bw = mSpatialWeight.weight<BandwidthWeight>()->bandwidth();
        if (!(bw > 0))
        {
            return false;
        }

        return true;
    }
    else return false;
}
