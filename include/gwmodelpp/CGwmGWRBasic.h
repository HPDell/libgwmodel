#ifndef CGWMGWRBASIC_H
#define CGWMGWRBASIC_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"

using namespace std;

class CGwmGWRBasic : public CGwmGWRBase, public IGwmBandwidthSelectable, public IGwmVarialbeSelectable, public IGwmOpenmpParallelizable
{
public:
    enum BandwidthSelectionCriterionType
    {
        AIC,
        CV
    };

    enum NameFormat
    {
        Fixed,
        VarName,
        PrefixVarName,
        SuffixVariable
    };

    static unordered_map<BandwidthSelectionCriterionType, string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef mat (CGwmGWRBasic::*RegressionCalculator)(const mat&, const vec&);
    typedef mat (CGwmGWRBasic::*RegressionHatmatrixCalculator)(const mat&, const vec&, mat&, vec&, vec&, mat&);

    typedef tuple<string, mat, NameFormat> ResultLayerDataItem;

    typedef double (CGwmGWRBasic::*BandwidthSelectionCriterionCalculator)(CGwmBandwidthWeight*);
    typedef double (CGwmGWRBasic::*IndepVarsSelectCriterionCalculator)(const vector<GwmVariable>&);

private:
    static GwmRegressionDiagnostic CalcDiagnostic(const mat& x, const vec& y, const mat& betas, const vec& shat);

public:
    CGwmGWRBasic();
    ~CGwmGWRBasic();

public:
    bool isAutoselectBandwidth() const;
    void setIsAutoselectBandwidth(bool isAutoSelect);

    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const;
    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    bool isAutoselectIndepVars() const;
    void setIsAutoselectIndepVars(bool isAutoSelect);

    double indepVarSelectionThreshold() const;
    void setIndepVarSelectionThreshold(double threshold);
    
    VariablesCriterionList indepVarsSelectionCriterionList() const;
    BandwidthCriterionList bandwidthSelectionCriterionList() const;

    bool hasHatMatrix() const;
    void setHasHatMatrix(const bool has);

public:     // Implement CGwmAlgorithm
    void run() override;

public:     // Implement IGwmRegressionAnalysis
    mat regression(const mat& x, const vec& y) override;
    mat regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S) override;
private:
    mat regressionSerial(const mat& x, const vec& y);
    mat regressionHatmatrixSerial(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
    
#ifdef ENABLE_OPENMP
    mat regressionOmp(const mat& x, const vec& y);
    mat regressionHatmatrixOmp(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qDiag, mat& S);
#endif

public:     // Implement IGwmBandwidthSelectable
    double getCriterion(CGwmBandwidthWeight* weight);
private:
    double bandwidthSizeCriterionCVSerial(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICSerial(CGwmBandwidthWeight* bandwidthWeight);
#ifdef ENABLE_OPENMP
    double bandwidthSizeCriterionCVOmp(CGwmBandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICOmp(CGwmBandwidthWeight* bandwidthWeight);
#endif

public:     // Implement IGwmVariableSelectable
    double getCriterion(const vector<GwmVariable>& variables);
private:
    double indepVarsSelectionCriterionSerial(const vector<GwmVariable>& indepVars);
#ifdef ENABLE_OPENMP
    double indepVarsSelectionCriterionOmp(const vector<GwmVariable>& indepVars);
#endif    

public:     // Implement IGwmParallelizable
    int parallelAbility() const;
    ParallelType parallelType() const;
    void setParallelType(const ParallelType& type);

public:     // Implement IGwmOpenmpParallelizable
    void setOmpThreadNum(const int threadNum);

protected:
    bool isStoreS();

    void createRegressionDistanceParameter();
    void createPredictionDistanceParameter();

    void createResultLayer(initializer_list<ResultLayerDataItem> items);

protected:
    bool mHasHatMatrix = true;
    bool mHasFTest = false;
    bool mHasPredict = false;

    DistanceParameter* mRegressionDistanceParameter = nullptr;
    DistanceParameter* mPredictionDistanceParameter = nullptr;

    bool mIsAutoselectIndepVars = false;
    double mIndepVarSelectionThreshold = 3.0;
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &CGwmGWRBasic::indepVarsSelectionCriterionSerial;
    VariablesCriterionList mIndepVarsSelectionCriterionList;

    bool mIsAutoselectBandwidth = false;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &CGwmGWRBasic::bandwidthSizeCriterionCVSerial;
    BandwidthCriterionList mBandwidthSelectionCriterionList;

    RegressionCalculator mPredictFunction = &CGwmGWRBasic::regressionSerial;
    RegressionHatmatrixCalculator mRegressionHatmatrixFunction = &CGwmGWRBasic::regressionHatmatrixSerial;

    ParallelType mParallelType = ParallelType::SerialOnly;
    int mOmpThreadNum = 8;
};

inline mat CGwmGWRBasic::regression(const mat& x, const vec& y)
{
    return regressionSerial(x, y);
}

inline mat CGwmGWRBasic::regressionHatmatrix(const mat& x, const vec& y, mat& betasSE, vec& shat, vec& qdiag, mat& S)
{
    return regressionHatmatrixSerial(x, y, betasSE, shat, qdiag, S);
}

inline bool CGwmGWRBasic::isStoreS()
{
    return mHasHatMatrix && (mSourceLayer->featureCount() < 8192);
}

inline bool CGwmGWRBasic::isAutoselectBandwidth() const
{
    return mIsAutoselectBandwidth;
}

inline void CGwmGWRBasic::setIsAutoselectBandwidth(bool isAutoSelect)
{
    mIsAutoselectBandwidth = isAutoSelect;
}

inline bool CGwmGWRBasic::isAutoselectIndepVars() const
{
    return mIsAutoselectIndepVars;
}

inline void CGwmGWRBasic::setIsAutoselectIndepVars(bool isAutoSelect)
{
    mIsAutoselectIndepVars = isAutoSelect;
}

inline double CGwmGWRBasic::indepVarSelectionThreshold() const
{
    return mIndepVarSelectionThreshold;
}

inline void CGwmGWRBasic::setIndepVarSelectionThreshold(double threshold)
{
    mIndepVarSelectionThreshold = threshold;
}

inline CGwmGWRBasic::BandwidthSelectionCriterionType CGwmGWRBasic::bandwidthSelectionCriterion() const
{
    return mBandwidthSelectionCriterion;
}
    
inline VariablesCriterionList CGwmGWRBasic::indepVarsSelectionCriterionList() const
{
    return mIndepVarsSelectionCriterionList;
}

inline BandwidthCriterionList CGwmGWRBasic::bandwidthSelectionCriterionList() const
{
    return mBandwidthSelectionCriterionList;
}

inline double CGwmGWRBasic::getCriterion(CGwmBandwidthWeight* weight)
{
    return (this->*mBandwidthSelectionCriterionFunction)(weight);
}

inline double CGwmGWRBasic::getCriterion(const vector<GwmVariable>& variables)
{
    return (this->*mIndepVarsSelectionCriterionFunction)(variables);
}

inline int CGwmGWRBasic::parallelAbility() const
{
    return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
#endif        
        ;
}

inline ParallelType CGwmGWRBasic::parallelType() const
{
    return mParallelType;
}

inline void CGwmGWRBasic::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

inline bool CGwmGWRBasic::hasHatMatrix() const
{
    return mHasHatMatrix;
}

inline void CGwmGWRBasic::setHasHatMatrix(const bool has)
{
    mHasHatMatrix = has;
}

#endif  // CGWMGWRBASIC_H