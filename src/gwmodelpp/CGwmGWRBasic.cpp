#include "CGwmGWRBasic.h"
#include "CGwmBandwidthSelector.h"
#include "CGwmVariableForwardSelector.h"
#include <assert.h>

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

using namespace arma;

GwmRegressionDiagnostic CGwmGWRBasic::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
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

CGwmGWRBasic::CGwmGWRBasic()
{
    
}

CGwmGWRBasic::~CGwmGWRBasic()
{
    
}

void CGwmGWRBasic::run()
{
    createRegressionDistanceParameter();
    assert(mRegressionDistanceParameter != nullptr);

    if (!hasPredictLayer() && mIsAutoselectIndepVars)
    {
        CGwmVariableForwardSelector selector(mIndepVars, mIndepVarSelectionThreshold);
        vector<GwmVariable> selectedIndepVars = selector.optimize(this);
        if (selectedIndepVars.size() > 0)
        {
            mIndepVars = selectedIndepVars;
            mIndepVarsSelectionCriterionList = selector.indepVarsCriterion();
        }
    }

    setXY(mX, mY, mSourceLayer, mDepVar, mIndepVars);
    uword nDp = mSourceLayer->featureCount();

    if (!hasPredictLayer() && mIsAutoselectBandwidth)
    {
        CGwmBandwidthWeight* bw0 = mSpatialWeight.weight<CGwmBandwidthWeight>();
        double lower = bw0->adaptive() ? 20 : 0.0;
        double upper = bw0->adaptive() ? nDp : mSpatialWeight.distance()->maxDistance(nDp, mRegressionDistanceParameter);
        CGwmBandwidthSelector selector(bw0, lower, upper);
        CGwmBandwidthWeight* bw = selector.optimize(this);
        if (bw)
        {
            mSpatialWeight.setWeight(bw);
            mBandwidthSelectionCriterionList = selector.bandwidthCriterion();
        }
    }

    if (mHasHatMatrix)
    {
        mat betasSE, S;
        vec shat, qdiag;
        mBetas = regressionHatmatrix(mX, mY, betasSE, shat, qdiag, S);
        mDiagnostic = CalcDiagnostic(mX, mY, mBetas, shat);
        double trS = shat(0), trStS = shat(1);
        double sigmaHat = mDiagnostic.RSS / (nDp - 2 * trS + trStS);
        betasSE = sqrt(sigmaHat * betasSE);
        vec yhat = Fitted(mX, mBetas);
        vec res = mY - yhat;
        vec stu_res = res / sqrt(sigmaHat * qdiag);
        mat betasTV = mBetas / betasSE;
        vec dybar2 = (mY - mean(mY)) % (mY - mean(mY));
        vec dyhat2 = (mY - yhat) % (mY - yhat);
        vec localR2 = vec(nDp, fill::zeros);
        for (uword i = 0; i < nDp; i++)
        {
            vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter, i);
            double tss = sum(dybar2 % w);
            double rss = sum(dyhat2 % w);
            localR2(i) = (tss - rss) / tss;
        }
        createResultLayer({
            make_tuple(string("%1"), mBetas, NameFormat::VarName),
            make_tuple(string("y"), mY, NameFormat::Fixed),
            make_tuple(string("yhat"), yhat, NameFormat::Fixed),
            make_tuple(string("residual"), res, NameFormat::Fixed),
            make_tuple(string("Stud_residual"), stu_res, NameFormat::Fixed),
            make_tuple(string("SE"), betasSE, NameFormat::PrefixVarName),
            make_tuple(string("TV"), betasTV, NameFormat::PrefixVarName),
            make_tuple(string("localR2"), localR2, NameFormat::Fixed)
        });
    }
    else
    {
        createPredictionDistanceParameter();
        mBetas = regression(mX, mY);
        createResultLayer({
            make_tuple(string("%1"), mBetas, NameFormat::VarName)
        });
    }
}

void CGwmGWRBasic::createRegressionDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mRegressionDistanceParameter = new CRSDistanceParameter(mSourceLayer->points(), mSourceLayer->points());
    }
}

void CGwmGWRBasic::createPredictionDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mPredictionDistanceParameter = new CRSDistanceParameter(hasPredictLayer() ? mPredictLayer->points() : mSourceLayer->points(), mSourceLayer->points());
    }
}

mat CGwmGWRBasic::regressionSerial(const mat& x, const vec& y)
{
    uword nRp = hasPredictLayer() ? mPredictLayer->featureCount() : mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        vec w = mSpatialWeight.weightVector(mPredictionDistanceParameter, i);
        mat xtw = trans(x.each_col() % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
        }
        catch (exception e)
        {
            throw e;
        }
    }
    return betas.t();
}

mat CGwmGWRBasic::regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    shat = vec(2, fill::zeros);
    qDiag = vec(nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter, i);
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
        catch (std::exception e)
        {
            throw e;
        }
    }
    betasSE = betasSE.t();
    return betas.t();
}

#ifdef ENABLE_OPENMP
mat CGwmGWRBasic::regressionOmp(const mat& x, const vec& y)
{
    uword nRp = hasPredictLayer() ? mPredictLayer->featureCount() : mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nRp, arma::fill::zeros);
    int current = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nRp; i++)
    {
        int thread = omp_get_thread_num();
        vec w = mSpatialWeight.weightVector(mPredictionDistanceParameter, i);
        mat xtw = trans(x.each_col() % w);
        mat xtwx = xtw * x;
        mat xtwy = xtw * y;
        try
        {
            mat xtwx_inv = inv_sympd(xtwx);
            betas.col(i) = xtwx_inv * xtwy;
        }
        catch (exception e)
        {
            throw e;
        }
    }
    return betas.t();
}

mat CGwmGWRBasic::regressionHatmatrixOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    betasSE = mat(nVar, nDp, fill::zeros);
    S = mat(isStoreS() ? nDp : 1, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    mat qDiag_all(nDp, mOmpThreadNum, fill::zeros);
    int current = 0;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        int thread = omp_get_thread_num();
        vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter, i);
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
        catch (std::exception e)
        {
            throw e;
        }
    }
    shat = sum(shat_all, 1);
    qDiag = sum(qDiag_all, 1);
    betasSE = betasSE.t();
    return betas.t();
}
#endif

double CGwmGWRBasic::bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight)
{
    uword nDp = mSourceLayer->featureCount();
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        vec d = mSpatialWeight.distance()->distance(mRegressionDistanceParameter, i);
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
        catch (...)
        {
            return DBL_MAX;
        }
    }
    if (isfinite(cv))
    {
        return cv;
    }
    else return DBL_MAX;
}

double CGwmGWRBasic::bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
        vec d = mSpatialWeight.distance()->distance(mRegressionDistanceParameter, i);
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
        catch (std::exception e)
        {
            return DBL_MAX;
        }
    }
    double value = CGwmGWRBase::AICc(mX, mY, betas.t(), shat);
    if (isfinite(value))
    {
        return value;
    }
    else return DBL_MAX;
}

#ifdef ENABLE_OPENMP
double CGwmGWRBasic::bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight)
{
    uword nDp = mSourceLayer->featureCount();
    vec shat(2, fill::zeros);
    vec cv_all(mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeight.distance()->distance(mRegressionDistanceParameter, i);
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
            catch (...)
            {
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

double CGwmGWRBasic::bandwidthSizeCriterionAICOmp(CGwmBandwidthWeight* bandwidthWeight)
{
    uword nDp = mSourceLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    mat shat_all(2, mOmpThreadNum, fill::zeros);
    bool flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
        if (flag)
        {
            int thread = omp_get_thread_num();
            vec d = mSpatialWeight.distance()->distance(mRegressionDistanceParameter, i);
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
            catch (std::exception e)
            {
                flag = false;
            }
        }
    }
    if (flag)
    {
        vec shat = sum(shat_all, 1);
        double value = CGwmGWRBase::AICc(mX, mY, betas.t(), shat);
        if (isfinite(value))
        {
            return value;
        }
        else return DBL_MAX;
    }
    else return DBL_MAX;
}
#endif

double CGwmGWRBasic::indepVarsSelectionCriterionSerial(const vector<GwmVariable>& indepVars)
{
    mat x;
    vec y;
    setXY(x, y, mSourceLayer, mDepVar, indepVars);
    uword nDp = x.n_rows, nVar = x.n_cols;
    mat betas(nVar, nDp, fill::zeros);
    vec shat(2, fill::zeros);
    for (uword i = 0; i < nDp; i++)
    {
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
            shat(0) += si(0, i);
            shat(1) += det(si * si.t());
        }
        catch (...)
        {
            return DBL_MAX;
        }
    }
    double value = CGwmGWRBase::AICc(x, y, betas.t(), shat);
    string msg = "Model: " + mDepVar.name + " ~ ";
    for (size_t i = 0; i < indepVars.size() - 1; i++)
    {
        msg += indepVars[i].name + " + ";
    }
    msg += indepVars.back().name;
    msg += " (AICc Value: " + to_string(value) + ")";
    
    return value;
}

#ifdef ENABLE_OPENMP
double CGwmGWRBasic::indepVarsSelectionCriterionOmp(const vector<GwmVariable>& indepVars)
{
    mat x;
    vec y;
    setXY(x, y, mSourceLayer, mDepVar, indepVars);
    uword nDp = mSourceLayer->featureCount(), nVar = indepVars.size() + 1;
    mat betas(nVar, nDp, fill::zeros);
    mat shat(2, mOmpThreadNum, fill::zeros);
    int flag = true;
#pragma omp parallel for num_threads(mOmpThreadNum)
    for (int i = 0; (uword)i < nDp; i++)
    {
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
            catch (...)
            {
                flag = false;
            }
        }
    }
    if (flag)
    {
        double value = CGwmGWRBase::AICc(x, y, betas.t(), sum(shat, 1));
        
        return value;
    }
    else return DBL_MAX;
}
#endif

void CGwmGWRBasic::createResultLayer(initializer_list<ResultLayerDataItem> items)
{
    mat layerPoints = hasPredictLayer() ? mPredictLayer->points() : mSourceLayer->points();
    uword layerFeatureCount = layerPoints.n_rows;
    mat layerData(layerFeatureCount, 0);
    vector<string> layerFields;
    for (auto &&i : items)
    {
        NameFormat nf = get<2>(i);
        mat column = get<1>(i);
        string name = get<0>(i);
        if (nf == NameFormat::Fixed)
        {
            layerData = join_rows(layerData, column.col(0));
            layerFields.push_back(name);
        }
        else
        {
            layerData = join_rows(layerData, column);
            for (size_t k = 0; k < column.n_cols; k++)
            {
                string variableName = k == 0 ? "Intercept" : mIndepVars[k - 1].name;
                string fieldName;
                switch (nf)
                {
                case NameFormat::VarName:
                    fieldName = variableName;
                    break;
                case NameFormat::PrefixVarName:
                    fieldName = variableName + "_" + name;
                    break;
                case NameFormat::SuffixVariable:
                    fieldName = name + "_" + variableName;
                    break;
                default:
                    fieldName = variableName;
                }
                layerFields.push_back(fieldName);
            }
        }
        
    }
    
    mResultLayer = new CGwmSimpleLayer(layerPoints, layerData, layerFields);
}

void CGwmGWRBasic::setBandwidthSelectionCriterion(BandwidthSelectionCriterionType type)
{
    mBandwidthSelectionCriterion = type;
    unordered_map<BandwidthSelectionCriterionType, BandwidthSelectionCriterionCalculator> mapper = {
        make_pair(BandwidthSelectionCriterionType::CV, &CGwmGWRBasic::bandwidthSizeCriterionCVSerial),
        make_pair(BandwidthSelectionCriterionType::AIC, &CGwmGWRBasic::bandwidthSizeCriterionAICSerial)
    };
    mBandwidthSelectionCriterionFunction = mapper[mBandwidthSelectionCriterion];
}

void CGwmGWRBasic::setParallelType(const ParallelType& type)
{
    if (type & parallelAbility())
    {
        mParallelType = type;
        switch (type) {
        case ParallelType::SerialOnly:
            mPredictFunction = &CGwmGWRBasic::regressionSerial;
            mRegressionHatmatrixFunction = &CGwmGWRBasic::regressionHatmatrixSerial;
            mIndepVarsSelectionCriterionFunction = &CGwmGWRBasic::indepVarsSelectionCriterionSerial;
            break;
        case ParallelType::OpenMP:
            mPredictFunction = &CGwmGWRBasic::regressionOmp;
            mRegressionHatmatrixFunction = &CGwmGWRBasic::regressionHatmatrixOmp;
            mIndepVarsSelectionCriterionFunction = &CGwmGWRBasic::indepVarsSelectionCriterionOmp;
            break;
        default:
            mPredictFunction = &CGwmGWRBasic::regressionSerial;
            mRegressionHatmatrixFunction = &CGwmGWRBasic::regressionHatmatrixSerial;
            mIndepVarsSelectionCriterionFunction = &CGwmGWRBasic::indepVarsSelectionCriterionSerial;
            break;
        }
    }
    setBandwidthSelectionCriterion(mBandwidthSelectionCriterion);
}