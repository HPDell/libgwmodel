#include "CGwmGWRBasic.h"
#include "CGwmBandwidthSelector.h"
#include "CGwmVariableForwardSelector.h"

GwmRegressionDiagnostic CGwmGWRBasic::CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat)
{
    vec r = y - sum(betas % x, 1);
    double rss = sum(r % r);
    double n = x.n_rows;
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
    _ASSERT(mRegressionDistanceParameter != nullptr);

    if (!hasPredictLayer() && mIsAutoselectIndepVars)
    {
        CGwmVariableForwardSelector selector(mIndepVars, mIndepVarSelectionThreshold);
        vector<GwmVariable> selectedIndepVars = selector.optimize(this);
        if (selectedIndepVars.size() > 0)
        {
            mIndepVars = selectedIndepVars;
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
            vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter);
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
        mRegressionDistanceParameter = new CRSDistanceParameter(mSourceLayer->data(), mSourceLayer->data());
    }
}

void CGwmGWRBasic::createPredictionDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mRegressionDistanceParameter = new CRSDistanceParameter(mPredictLayer->data(), mSourceLayer->data());
    }
}

mat CGwmGWRBasic::regressionSerial(const mat& x, const vec& y)
{
    uword nRp = mPredictLayer->featureCount(), nVar = mIndepVars.size() + 1;
    mat betas(nVar, nRp, fill::zeros);
    for (uword i = 0; i < nRp; i++)
    {
        vec w = mSpatialWeight.weightVector(mPredictionDistanceParameter);
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
        vec w = mSpatialWeight.weightVector(mRegressionDistanceParameter);
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

double CGwmGWRBasic::bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight)
{
    uword nDp = mSourceLayer->featureCount();
    vec shat(2, fill::zeros);
    double cv = 0.0;
    for (uword i = 0; i < nDp; i++)
    {
        vec d = mSpatialWeight.distance()->distance(mRegressionDistanceParameter);
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
        vec d = mSpatialWeight.distance()->distance(mRegressionDistanceParameter);
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
    
    CGwmSimpleLayer* resultLayer = new CGwmSimpleLayer(layerPoints, layerData, layerFields);
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