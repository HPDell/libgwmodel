#ifndef CGWMGWDR_H
#define CGWMGWDR_H

#include <vector>
#include <armadillo>
#include <gsl/gsl_vector.h>
#include "CGwmSpatialAlgorithm.h"
#include "spatialweight/CGwmSpatialWeight.h"
#include "IGwmRegressionAnalysis.h"

using namespace std;
using namespace arma;


class CGwmGWDR : public CGwmSpatialAlgorithm, public IGwmRegressionAnalysis
{
public:
    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    enum BandwidthCriterionType
    {
        CV,
        AIC
    };

    typedef double (CGwmGWDR::*BandwidthCriterionCalculator)(const vector<CGwmBandwidthWeight*>&);

public:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

    static vec Fitted(const mat& x, const mat& betas)
    {
        return sum(betas % x, 1);
    }

    static double RSS(const mat& x, const mat& y, const mat& betas)
    {
        vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    static double AICc(const mat& x, const mat& y, const mat& betas, const vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

public:
    mat betas() const
    {
        return mBetas;
    }

    bool hasHatMatrix() const
    {
        return mHasHatMatrix;
    }

    void setHasHatMatrix(bool flag)
    {
        mHasHatMatrix = flag;
    }

public: // CGwmAlgorithm
    void run();
    bool isValid()
    {
        // [TODO]: Add actual check codes.
        return true;
    }

public: // IGwmRegressionAnalysis
    GwmVariable dependentVariable() const
    {
        return mDepVar;
    }

    void setDependentVariable(const GwmVariable& variable)
    {
        mDepVar = variable;
    }

    vector<GwmVariable> independentVariables() const
    {
        return mIndepVars;
    }

    void setIndependentVariables(const vector<GwmVariable>& variables)
    {
        mIndepVars = variables;
    }

    BandwidthCriterionType bandwidthCriterionType() const
    {
        return mBandwidthCriterionType;
    }

    void setBandwidthCriterionType(const BandwidthCriterionType& type);

public:  // IRgressionAnalysis

    mat regression(const mat& x, const vec& y)
    {
        return regressionSerial(x, y);
    }

    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
    {
        return regressionHatmatrixSerial(x, y, betasSE, shat, qdiag, S);
    }

    GwmRegressionDiagnostic diagnostic() const
    {
        return mDiagnostic;
    }

public:
    vector<CGwmSpatialWeight> spatialWeights()
    {
        return mSpatialWeights;
    }

    void setSpatialWeights(vector<CGwmSpatialWeight> spatialWeights)
    {
        mSpatialWeights = spatialWeights;
    }

    bool enableBandwidthOptimize()
    {
        return mEnableBandwidthOptimize;
    }

    void setEnableBandwidthOptimize(bool flag)
    {
        mEnableBandwidthOptimize = flag;
    }

public:
    double bandwidthCriterion(const vector<CGwmBandwidthWeight*>& bandwidths)
    {
        return (this->*mBandwidthCriterionFunction)(bandwidths);
    }

protected:
    mat regressionSerial(const mat& x, const vec& y);
    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S);

    double bandwidthCriterionCVSerial(const vector<CGwmBandwidthWeight*>& bandwidths);
    double bandwidthCriterionAICSerial(const vector<CGwmBandwidthWeight*>& bandwidths);

protected:
    void createResultLayer(initializer_list<ResultLayerDataItem> items);

private:
    void setXY(mat& x, mat& y, const CGwmSimpleLayer* layer, const GwmVariable& depVar, const vector<GwmVariable>& indepVars);

    bool isStoreS()
    {
        return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
    }

private:
    vector<CGwmSpatialWeight> mSpatialWeights;
    vector<DistanceParameter*> mDistParameters;

    vec mY;
    mat mX;
    mat mBetas;
    GwmVariable mDepVar;
    vector<GwmVariable> mIndepVars;
    bool mHasHatMatrix;
    GwmRegressionDiagnostic mDiagnostic;

    bool mEnableBandwidthOptimize = false;
    BandwidthCriterionType mBandwidthCriterionType = BandwidthCriterionType::CV;
    BandwidthCriterionCalculator mBandwidthCriterionFunction = &CGwmGWDR::bandwidthCriterionCVSerial;
};


class CGwmGWDRBandwidthOptimizer
{
public:
    struct Parameter
    {
        CGwmGWDR* instance;
        vector<CGwmBandwidthWeight*>* bandwidths;
        uword featureCount;
    };

    static double criterion_function(const gsl_vector* bws, void* params);

public:
    CGwmGWDRBandwidthOptimizer(vector<CGwmBandwidthWeight*> weights)
    {
        mBandwidths = weights;
    }

    const int optimize(CGwmGWDR* instance, uword featureCount, size_t maxIter, double eps);

private:
    vector<CGwmBandwidthWeight*> mBandwidths;
    Parameter mParameter;
};

#endif  // CGWMGWDR_H