#ifndef GWRLOCALCOLLINEARITY_H
#define GWRLOCALCOLLINEARITY_H

#include <utility>
#include <string>
#include <initializer_list>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"
#include <exception>


namespace gwm
{

/**
 * @brief \~english GWR model for dat of local collinearity. \~chinese 局部岭回归地理加权模型
 * 
 */
class GWRLocalCollinearity : public GWRBase, public IBandwidthSelectable, public IParallelizable, public IParallelOpenmpEnabled
{
public:

    /**
     * @brief \~english Type of bandwidth criterion. \~chinese 带宽优选指标值类型。
     * 
     */
    enum BandwidthSelectionCriterionType
    {
        CV     //!< CV
    };
    
    typedef double (GWRLocalCollinearity::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);    //!< \~english Calculator to get criterion for bandwidth optimization \~chinese 带宽优选指标值计算函数
    typedef arma::mat (GWRLocalCollinearity::*FitCalculator)(const arma::mat&, const arma::vec&);   //!< \~english Calculator to predict \~chinese 用于预测的函数
    typedef arma::mat (GWRLocalCollinearity::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);   //!< \~english Calculator to predict \~chinese 用于预测的函数

    /**
     * @brief \~english Calculate diagnostic information. \~chinese 计算诊断信息。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betas \~english Coefficient estimates \~chinese 回归系数估计值
     * @param shat \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @return RegressionDiagnostic \~english Diagnostic information \~chinese 诊断信息
     */
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:

    /**
     * @brief \~english Construct a new GWRLocalCollinearity object. \~chinese 构造一个新的 GWRLocalCollinearity 对象。
     * 
     */
    GWRLocalCollinearity();

    /**
     * @brief \~english Destroy the GWRLocalCollinearity object. \~chinese 销毁 GWRLocalCollinearity 对象。
     * 
     */
    ~GWRLocalCollinearity();

public:

    /**
     * @brief \~english Get the threshold. \~chinese 获取阈值
     * 
     * @return double \~english The threshold \~chinese 阈值
     */
    double cnThresh() const
    {
        return mCnThresh;
    }

    /**
     * @brief \~english Set the threshold. \~chinese 设置阈值
     * 
     * @param cnThresh \~english The threshold \~chinese 阈值
     */
    void setCnThresh(double cnThresh)
    {
        mCnThresh = cnThresh;
    }

    /**
     * @brief \~english Get the lambda value \~chinese 获取参数 lambda 的值
     * 
     * @return double \~english The lambda value \~chinese 参数 lambda 的值
     */
    double lambda() const
    {
        return mLambda;
    }

    /**
     * @brief \~english Set the lambda value \~chinese 设置参数 lambda 的值
     * 
     * @param lambda \~english The lambda value \~chinese 参数 lambda 的值
     */
    void setLambda(double lambda)
    {
        mLambda = lambda;
    }

    /**
     * @brief \~english Get whether has hat matrix. \~chinese 获取是否有帽子矩阵。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool hasHatMatrix() const
    {
        return mHasHatMatrix;
    }

    /**
     * @brief \~english Set whether has hat matrix. \~chinese 设置是否有帽子矩阵。 
     * 
     * @param flag \~english Whether has hat matrix. \~chinese 是否有帽子矩阵。 
     */
    void setHasHatMatrix(bool value)
    {
        mHasHatMatrix = value;
    }

    /**
     * @brief \~english Get whether to adjust lambda \~chinese 获取是否调整 lambda 值。
     * 
     * @return true \~english  \~chinese 
     * @return false \~english  \~chinese 
     */
    bool lambdaAdjust() const
    {
        return mLambdaAdjust;
    }

    /**
     * @brief \~english Set whether to adjust lambda \~chinese 设置是否调整 lambda 值。
     * 
     * @param lambdaAdjust \~english Whether to adjust lambda \~chinese 是否调整 lambda 值
     */
    void setLambdaAdjust(bool lambdaAdjust)
    {
        mLambdaAdjust = lambdaAdjust;
    }

    RegressionDiagnostic dialnostic() const
    {
        return mDiagnostic;
    }

    /**
     * @brief \~english Get whether bandwidth optimization is enabled. \~chinese 获取是否进行带宽优选。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool isAutoselectBandwidth() const
    {
        return mIsAutoselectBandwidth;
    }

    /**
     * @brief \~english Set whether bandwidth optimization is enabled. \~chinese 设置是否进行带宽优选。 
     * 
     * @param flag \~english Whether bandwidth optimization is enabled \~chinese 是否进行带宽优选
     */
    void setIsAutoselectBandwidth(bool isAutoSelect)
    {
        mIsAutoselectBandwidth = isAutoSelect;
    }

    /**
     * @brief \~english Get the list of criterion values for each bandwidth value. \~chinese 获取每种带宽对应的指标值列表。
     * 
     * @return BandwidthCriterionList \~english List of criterion values for each bandwidth value \~chinese 每种带宽对应的指标值列表
     */
    const BandwidthCriterionList& bandwidthSelectionCriterionList() const
    {
        return mBandwidthSelectionCriterionList;
    }

    /**
     * @brief \~english Get the type of criterion for bandwidth optimization. \~chinese 获取带宽优选指标类型。
     * 
     * @return BandwidthCriterionType \~english Type of criterion for bandwidth optimization \~chinese 带宽优选指标类型
     */
    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const
    {
        return mBandwidthSelectionCriterion;
    }

    /**
     * @brief \~english Set the type of criterion for bandwidth optimization. \~chinese 设置带宽优选指标类型。
     * 
     * @param type \~english Type of criterion for bandwidth optimization \~chinese 带宽优选指标类型
     */
    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    Status getCriterion(BandwidthWeight *bandwidthWeight, double& criterion) override
    {
        criterion = (this->*mBandwidthSelectionCriterionFunction)(bandwidthWeight);
        return mStatus;
    }

public:
    arma::mat fit() override;

    arma::mat predict(const arma::mat &locations) override;

private:

    /**
     * \~english
     * @brief Create distance parameters for prediction.
     * 
     * @param locations Distance parameters for prediction.
     * 
     * \~chinese
     * @brief 生成用于预测的距离参数。
     * 
     * @param locations 用于预测的距离参数。
     * 
     */
    void createPredictionDistanceParameter(const arma::mat& locations);

    /**
     * @brief \~english Non-parallel implementation of fitting function. \~chinese 拟合函数的非并行实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     */
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y);

    /**
     * @brief \~english Multithreading implementation of fitting function. \~chinese 拟合函数的多线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     */
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y);

    /**
     * @brief \~english Non-parallel implementation of prediction function. \~chinese 预测函数的非并行实现。
     * 
     * @param locations \~english Locations to predict \~chinese 要预测的位置
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english Multithreading implementation of prediction function. \~chinese 预测函数的多线程实现。
     * 
     * @param locations \~english Locations to predict \~chinese 要预测的位置
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
#endif

public:
    int parallelAbility() const override;
    ParallelType parallelType() const override;
    void setParallelType(const ParallelType& type) override;
    void setOmpThreadNum(const int threadNum) override;

protected:
    BandwidthCriterionList mBandwidthSelectionCriterionList;
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::CV;
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GWRLocalCollinearity::bandwidthSizeCriterionCVSerial;
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。


    // //这个函数并没有完成，计算local cv的话，直接放在了循环里面算，所以需求似乎也不大
    // /**
    //  * @brief \~english Get the CV. \~chinese 返回cv的函数
    //  * 
    //  * @param bw \~english  \~chinese 
    //  * @param kernel \~english  \~chinese 
    //  * @param adaptive \~english  \~chinese 
    //  * @param lambda \~english  \~chinese 
    //  * @param lambdaAdjust \~english  \~chinese 
    //  * @param cnThresh \~english  \~chinese 
    //  * @return double \~english  \~chinese 
    //  */
    // double LcrCV(double bw,arma::uword kernel, bool adaptive,double lambda,bool lambdaAdjust,double cnThresh);
    
    /**
     * @brief \~english Ridge linear regression. \~chinese 岭回归。
     * 
     * @param w \~english  \~chinese 
     * @param lambda \~english  \~chinese 
     * @return arma::vec \~english  \~chinese 
     */
    arma::vec ridgelm(const arma::vec& w,double lambda);

private:

    /**
     * @brief \~english Non-parallel implementation of calculator to get CV criterion for given bandwidths. \~chinese 获取给定带宽值对应的CV值的非并行实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthSizeCriterionCVSerial(BandwidthWeight* bandwidthWeight);

#ifdef ENABLE_OPENMP
    /**
     * @brief \~english Multithreading implementation of calculator to get CV criterion for given bandwidths. \~chinese 获取给定带宽值对应的CV值的多线程实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthSizeCriterionCVOmp(BandwidthWeight* bandwidthWeight);
#endif

private:

    double mLambda = 0;
    bool mLambdaAdjust = false;
    double mCnThresh = 30;
    bool mHasHatMatrix = false;
    bool mIsAutoselectBandwidth = false;
    double mTrS = 0;
    double mTrStS = 0;
    // arma::vec mSHat;

    FitCalculator mFitFunction = &GWRLocalCollinearity::fitSerial;
    PredictCalculator mPredictFunction = &GWRLocalCollinearity::predictSerial;
    ParallelType mParallelType = ParallelType::SerialOnly;

    int mOmpThreadNum = 8;
    arma::uword mGpuId = 0;
    arma::uword mGroupSize = 64;
};

inline int GWRLocalCollinearity::parallelAbility() const
{
    return ParallelType::SerialOnly | ParallelType::OpenMP;
}

inline ParallelType GWRLocalCollinearity::parallelType() const
{
    return mParallelType;
}

inline void GWRLocalCollinearity::setOmpThreadNum(const int threadNum)
{
    mOmpThreadNum = threadNum;
}

}

#endif  //GWRLOCALCOLLINEARITY_H