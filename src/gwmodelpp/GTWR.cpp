#include "GTWR.h"
#include "BandwidthSelector.h"
#include "VariableForwardSelector.h"
#include "Logger.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace arma;
using namespace gwm;

RegressionDiagnostic GTWR::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
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

mat GTWR::fit()
{
    createDistanceParameter();

    uword nDp = mCoords.n_rows;

    if (mIsAutoselectBandwidth)
    {
        BandwidthWeight* bw0 = mSpatialWeight.weight<BandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance();
        BandwidthSelector selector(bw0, lower, upper);
        BandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
    }

    mBetas = (this->*mFitFunction)(mX, mY, mBetasSE, mSHat, mQDiag, mS);
    mDiagnostic = CalcDiagnostic(mX, mY, mBetas, mSHat);
    double trS = mSHat(0), trStS = mSHat(1);
    double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
    mBetasSE = sqrt(sigmaHat * mBetasSE);
    vec yhat = Fitted(mX, mBetas);
    vec res = mY - yhat;
    vec stu_res = res / sqrt(sigmaHat * mQDiag);
    mat betasTV = mBetas / mBetasSE;
    vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
    vec dyhat2 = (mY - yhat) % (mY - yhat);
    vec localR2 = vec(nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w = mSpatialWeight.weightVector(i);
        double tss = sum(dybar2 % w);
        double rss = sum(dyhat2 % w);
        localR2(i) = (tss - rss) / tss;
    }

    return mBetas;
}

mat GTWR::predict(const mat& locations)
{
    createPredictionDistanceParameter(locations);
    mBetas = (this->*mPredictFunction)(locations, mX, mY);
    return mBetas;
}

void GTWR::createPredictionDistanceParameter(const arma::mat& locations)
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSSTDistance)
    {
        mSpatialWeight.distance()->makeParameter({ mCoords, mCoords, vTimes, vTimes });
    }
}

void GTWR::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == Distance::DistanceType::CRSSTDistance)
    {
        mSpatialWeight.distance()->makeParameter({ mCoords, mCoords, vTimes, vTimes });
    }
}


mat GTWR::predictSerial(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
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
            throw e;
        }
    }
    return betas.t();
}

mat GTWR::fitSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mCoords.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    qDiag = vec(nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
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
    }
    betasSE = betasSE.t();
    return betas.t();
}

double GTWR::bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
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
            cv += res * res;
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    if (isfinite(cv))
    {
        return cv;
    }
    else return DBL_MAX;
}

double GTWR::bandwidthSizeCriterionAICSerial(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
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
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (const exception& e)
        {
            GWM_LOG_ERROR(e.what());
            return DBL_MAX;
        }
    }
    double value = GWRBase::AICc(mX, mY, betas.t(), shat);
    if (isfinite(value))
    {
        return value;
    }
    else return DBL_MAX;
}

// double GTWR::indepVarsSelectionCriterionSerial(const vector<size_t>& indepVars)
// {
//     mat x = mX.cols(VariableForwardSelector::index2uvec(indepVars, mHasIntercept));
//     vec y = mY;
//     uword nDp = mCoords.n_rows, nVar = x.n_cols;
//     mat betas(nVar, nDp, fill::zeros);
//     vec shat(2, fill::zeros);
//     for (uword i = 0; i < nDp; i++)
//     {
//         vec w(nDp, fill::ones);
//         mat xtw = trans(x.each_col() % w);
//         mat xtwx = xtw * x;
//         mat xtwy = xtw * y;
//         try
//         {
//             mat xtwx_inv = inv_sympd(xtwx);
//             betas.col(i) = xtwx_inv * xtwy;
//             mat ci = xtwx_inv * xtw;
//             mat si = x.row(i) * ci;
//             shat(0) += si(0, i);
//             shat(1) += det(si * si.t());
//         }
//         catch (const exception& e)
//         {
//             GWM_LOG_ERROR(e.what());
//             return DBL_MAX;
//         }
//     }
//     double value = GWRBase::AICc(x, y, betas.t(), shat);    
//     return value;
// }

void GTWR::setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion)
{
    mBandwidthSelectionCriterion = criterion;
    unordered_map<BandwidthSelectionCriterionType, BandwidthSelectionCriterionCalculator> mapper;
    mapper = {
        make_pair(BandwidthSelectionCriterionType::CV, &GTWR::bandwidthSizeCriterionCVSerial),
        make_pair(BandwidthSelectionCriterionType::AIC, &GTWR::bandwidthSizeCriterionAICSerial)
    };
    mBandwidthSelectionCriterionFunction = mapper[mBandwidthSelectionCriterion];
}

bool GTWR::isValid()
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

#ifdef ENABLE_OPENMP
mat GTWR::predictOmp(const mat& locations, const mat& x, const vec& y)
{
    uword nRp = locations.n_rows, nVar = x.n_cols;
    mat betas(nVar, nRp, arma::fill::zeros);
    bool success = true;
    std::exception except;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nRp; i++)
    {
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
    }
    if (!success)
    {
        throw except;
    }
    return betas.t();
}

mat GTWR::fitOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
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
#endif

#ifdef ENABLE_OPENMP
double GTWR::bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows;
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
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
    if (flag)
    {
        double cv = sum(cv_all);
        return cv;
    }
    else return DBL_MAX;
}

double GTWR::bandwidthSizeCriterionAICOmp(BandwidthWeight* bandwidthWeight)
{
    uword nDp = mCoords.n_rows, nVar = mX.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
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
    if (flag)
    {
        vec shat = sum(shat_all, 1);
        double value = GWRBase::AICc(mX, mY, betas.t(), shat);
        if (isfinite(value))
        {
            return value;
        }
        else return DBL_MAX;
    }
    else return DBL_MAX;
}
#endif

void GTWR::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &GTWR::predictSerial;
            mFitFunction = &GTWR::fitSerial;
            break;
#ifdef ENABLE_OPENMP
        case ParallelType::OpenMP:
            mPredictFunction = &GTWR::predictOmp;
            mFitFunction = &GTWR::fitOmp;
            break;
#endif
        default:
            mPredictFunction = &GTWR::predictSerial;
            mFitFunction = &GTWR::fitSerial;
            break;
        }
        setBandwidthSelectionCriterion(mBandwidthSelectionCriterion);
    }
}