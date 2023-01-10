#ifndef CGWMROBUSTGWR_H
#define CGWMROBUSTGWR_H

#include <utility>
#include <string>
#include <initializer_list>
#include "CGwmGWRBase.h"
#include "GwmRegressionDiagnostic.h"
#include "IGwmBandwidthSelectable.h"
#include "IGwmVarialbeSelectable.h"
#include "IGwmParallelizable.h"
#include "CGwmGWRBasic.h"

class CGwmGWRRobust : public CGwmGWRBasic
{
private:
    typedef arma::mat (CGwmGWRRobust::*RegressionHatmatrix)(const arma::mat &, const arma::vec &, arma::mat &, arma::vec &, arma::vec &, arma::mat &);

    static GwmRegressionDiagnostic CalcDiagnostic(const arma::mat &x, const arma::vec &y, const arma::mat &betas, const arma::vec &shat);

public:
    CGwmGWRRobust();
    ~CGwmGWRRobust();

public:
    bool filtered() const;
    void setFiltered(bool value);

public: // Implement IGwmRegressionAnalysis
    //arma::mat regression(const arma::mat &x, const arma::vec &y) override;
    arma::mat predict(const arma::mat& locations) override;
    arma::mat fit() override;
    arma::mat regressionHatmatrix(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qdiag, arma::mat &S);

private:
    //arma::mat regressionHatmatrixSerial(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    //arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
       
#ifdef ENABLE_OPENMP
    //arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);
#endif

public: // Implement CGwmAlgorithm
    //void run();

protected:
    arma::mat robustGWRCaliFirst(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    // 第二种解法
    arma::mat robustGWRCaliSecond(const arma::mat &x, const arma::vec &y, arma::mat &betasSE, arma::vec &shat, arma::vec &qDiag, arma::mat &S);
    // 计算二次权重函数
    arma::vec filtWeight(arma::vec residual, double mse);

public : // Implement IGwmParallelizable
    void setParallelType(const ParallelType &type) override;

protected:
    void createPredictionDistanceParameter(const arma::mat& locations);

private:
    bool mFiltered;

    arma::mat mS;
    arma::vec mWeightMask;
    
    RegressionHatmatrix mfitFunction = &CGwmGWRRobust::fitSerial;
    
};

inline bool CGwmGWRRobust::filtered() const
{
    return mFiltered;
}

inline void CGwmGWRRobust::setFiltered(bool value)
{
    if (value)
    {
        this->mFiltered = true;
    }
    else
    {
        this->mFiltered = false;
    }
}

#endif // CGWMROBUSTGWR_H