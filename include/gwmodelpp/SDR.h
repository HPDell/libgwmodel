#ifndef SDR_H
#define SDR_H

#include <vector>
#include <armadillo>
#include <gsl/gsl_vector.h>
#include "SpatialAlgorithm.h"
#include "spatialweight/SpatialWeight.h"
#include "IRegressionAnalysis.h"
#include "VariableForwardSelector.h"
#include "IParallelizable.h"
#include "IBandwidthSelectable.h"

namespace gwm
{

/**
 * @brief \~english Geographically Weighted Density Regression \~chinese 地理加权密度回归模型
 * 
 */
class SDR : public SpatialAlgorithm, public IRegressionAnalysis, public IVarialbeSelectable, public IParallelizable, public IParallelOpenmpEnabled
{
public:
    typedef arma::mat (SDR::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&); //!< \~english Calculator to predict \~chinese 用于预测的函数

    typedef arma::mat (SDR::*FitCalculator)(const arma::mat&, const arma::vec&, arma::mat&, arma::vec&, arma::vec&, arma::mat&);   //!< \~english Calculator to fit \~chinese 用于拟合的函数

    /**
     * @brief \~english Type of bandwidth criterion. \~chinese 带宽优选指标值类型。
     * 
     */
    enum BandwidthCriterionType
    {
        CV,     //!< CV
        AIC     //!< AIC
    };

    typedef double (SDR::*BandwidthCriterionCalculator)(const std::vector<BandwidthWeight*>&); //!< \~english Calculator to get criterion for bandwidth optimization \~chinese 带宽优选指标值计算函数

    typedef double (SDR::*IndepVarCriterionCalculator)(const std::vector<std::size_t>&); //!< \~english Calculator to get criterion for variable optimization \~chinese 变量优选指标值计算函数

public:
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

    /**
     * @brief \~english Get fitted value of dependent variable. \~chinese 获取因变量估计值。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param betas \~english Coefficient estimates \~chinese 回归系数估计值
     * @return arma::vec \~english Fitted value of dependent variable \~chinese 因变量估计值
     */
    static arma::vec Fitted(const arma::mat& x, const arma::mat& betas)
    {
        return sum(betas % x, 1);
    }

    /**
     * @brief \~english Get sum of squared residuals. \~chinese 获取残差平方和。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betas \~english Coefficient estimates \~chinese 回归系数估计值
     * @return double \~english Sum of squared residuals \~chinese 残差平方和
     */
    static double RSS(const arma::mat& x, const arma::mat& y, const arma::mat& betas)
    {
        arma::vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    /**
     * @brief \~english Get the corrected AIC value. \~chinese 获取 AICc 值。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betas \~english Coefficient estimates \~chinese 回归系数估计值
     * @param shat \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @return double \~english  \~chinese 
     */
    static double AICc(const arma::mat& x, const arma::mat& y, const arma::mat& betas, const arma::vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * arma::datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

public:

    /**
     * @brief \~english Construct a new SDR object. \~chinese 构造一个新的 SDR 对象。
     * 
     */
    SDR() {}

    /**
     * @brief \~english Construct a new SDR object. \~chinese 构造一个新的 SDR 对象。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param coords \~english Coordinates of samples \~chinese 样本坐标
     * @param spatialWeights \~english Spatial weighting scheme \~chinese 空间权重配置
     * @param hasHatMatrix \~english Whether has hat matrix \~chinese 是否计算帽子矩阵
     * @param hasIntercept \~english Whether has intercept \~chinese 是否包含截距
     */
    SDR(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const std::vector<SpatialWeight>& spatialWeights, bool hasHatMatrix = true, bool hasIntercept = true) : SpatialAlgorithm(coords),
        mX(x),
        mY(y),
        mSpatialWeights(spatialWeights),
        mHasHatMatrix(hasHatMatrix),
        mHasIntercept(hasIntercept)
    {
    }

    /**
     * @brief \~english Destroy the SDR object. \~chinese 销毁 SDR 对象。
     * 
     */
    virtual ~SDR() {}

public:

    /**
     * @brief \~english Get coefficient estimates. \~chinese 获取回归系数估计值。
     * 
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    const arma::mat& betas() const { return mBetas; }

    /**
     * @brief \~english Get whether has hat matrix. \~chinese 获取是否有帽子矩阵。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool hasHatMatrix() const { return mHasHatMatrix; }

    /**
     * @brief \~english Set whether has hat matrix. \~chinese 设置是否有帽子矩阵。 
     * 
     * @param flag \~english Whether has hat matrix. \~chinese 是否有帽子矩阵。 
     */
    void setHasHatMatrix(bool flag) { mHasHatMatrix = flag; }

    /**
     * @brief \~english Get spatial weighting scheme \~chinese 获取空间权重配置。
     * 
     * @return const std::vector<SpatialWeight>& \~english Spatial weighting scheme \~chinese 空间权重配置
     */
    const std::vector<SpatialWeight>& spatialWeights() const { return mSpatialWeights; }

    /**
     * @brief \~english Set spatial weighting scheme. \~chinese 设置空间权重配置。 
     * 
     * @param spatialWeights \~english Spatial weighting scheme \~chinese 空间权重配置
     */
    void setSpatialWeights(const std::vector<SpatialWeight>& spatialWeights) { mSpatialWeights = spatialWeights; }

    /**
     * @brief \~english Get whether bandwidth optimization is enabled. \~chinese 获取是否进行带宽优选。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool enableBandwidthOptimize() { return mEnableBandwidthOptimize; }

    /**
     * @brief \~english Set whether bandwidth optimization is enabled. \~chinese 设置是否进行带宽优选。 
     * 
     * @param flag \~english Whether bandwidth optimization is enabled \~chinese 是否进行带宽优选
     */
    void setEnableBandwidthOptimize(bool flag) { mEnableBandwidthOptimize = flag; }

    /**
     * @brief \~english Get the threshold for bandwidth optimization. \~chinese 获取带宽优选阈值。
     * 
     * @return double \~english Threshold for bandwidth optimization \~chinese 带宽优选阈值
     */
    double bandwidthOptimizeEps() const { return mBandwidthOptimizeEps; }

    /**
     * @brief \~english Set the threshold for bandwidth optimization. \~chinese 设置带宽优选阈值。 
     * 
     * @param value \~english Threshold for bandwidth optimization \~chinese 带宽优选阈值
     */
    void setBandwidthOptimizeEps(double value) { mBandwidthOptimizeEps = value; }

    /**
     * @brief \~english Get the maximum iteration for bandwidth optimization. \~chinese 获取带宽优选最大迭代次数。
     * 
     * @return std::size_t \~english Maximum iteration for bandwidth optimization \~chinese 带宽优选最大迭代次数
     */
    std::size_t bandwidthOptimizeMaxIter() const { return mBandwidthOptimizeMaxIter; }

    /**
     * @brief \~english Set the maximum iteration for bandwidth optimization. \~chinese 获取带宽优选最大迭代次数。
     * 
     * @param value \~english Maximum iteration for bandwidth optimization \~chinese 带宽优选最大迭代次数
     */
    void setBandwidthOptimizeMaxIter(std::size_t value) { mBandwidthOptimizeMaxIter = value; }

    /**
     * @brief \~english Get the step size for bandwidth optimization. \~chinese 获取带宽优选步长。
     * 
     * @return double \~english Step size for bandwidth optimization \~chinese 带宽优选步长
     */
    double bandwidthOptimizeStep() const { return mBandwidthOptimizeStep; }

    /**
     * @brief \~english Set the step size for bandwidth optimization. \~chinese 设置带宽优选步长。
     * 
     * @param value \~english Step size for bandwidth optimization \~chinese 带宽优选步长
     */
    void setBandwidthOptimizeStep(double value) { mBandwidthOptimizeStep = value; }

    /**
     * @brief \~english Get the type of criterion for bandwidth optimization. \~chinese 获取带宽优选指标类型。
     * 
     * @return BandwidthCriterionType \~english Type of criterion for bandwidth optimization \~chinese 带宽优选指标类型
     */
    BandwidthCriterionType bandwidthCriterionType() const { return mBandwidthCriterionType; }

    /**
     * @brief \~english Set the type of criterion for bandwidth optimization. \~chinese 设置带宽优选指标类型。
     * 
     * @param type \~english Type of criterion for bandwidth optimization \~chinese 带宽优选指标类型
     */
    void setBandwidthCriterionType(const BandwidthCriterionType& type);

    /**
     * @brief \~english Get whether independent variable selection is enabled. \~chinese 获取是否优选变量。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese 否
     */
    bool enableIndpenVarSelect() const { return mEnableIndepVarSelect; }

    /**
     * @brief \~english Set whether independent variable selection is enabled. \~chinese 设置是否优选变量。
     * 
     * @param flag \~english Whether independent variable selection is enabled \~chinese 是否优选变量
     */
    void setEnableIndepVarSelect(bool flag) { mEnableIndepVarSelect = flag; }

    /**
     * @brief \~english Get threshold for independent variable selection \~chinese 获取变量优选阈值
     * 
     * @return double \~english Threshold for independent variable selection \~chinese 变量优选阈值
     */
    double indepVarSelectThreshold() const { return mIndepVarSelectThreshold; }

    /**
     * @brief \~english Set threshold for independent variable selection \~chinese 设置变量优选阈值
     * 
     * @param threshold \~english Threshold for independent variable selection \~chinese 变量优选阈值
     */
    void setIndepVarSelectThreshold(double threshold) { mIndepVarSelectThreshold = threshold; }

    /**
     * @brief \~english Get the list of criterion values for each variable combination in independent variable selection. \~chinese 获取变量优选过程中每种变量组合对应的指标值列表。
     * 
     * @return VariablesCriterionList \~english List of criterion values for each variable combination in independent variable selection \~chinese 变量优选过程中每种变量组合对应的指标值列表
     */
    const VariablesCriterionList& indepVarCriterionList() const { return mIndepVarCriterionList; }

    /**
     * @brief \~english Get selected independent variable \~chinese 获取选中的变量组合
     * 
     * @return const std::vector<std::size_t>& \~english Selected independent variable \~chinese 选中的变量组合
     */
    const std::vector<std::size_t>& selectedIndepVars() const { return mSelectedIndepVars; }

    /**
     * @brief \~english Get the standard error of coefficient estimates. \~chinese 获取回归系数估计值标准误差。
     * 
     * @return arma::mat \~english Standard error of coefficient estimates \~chinese 回归系数估计值标准误差。
     */
    arma::mat betasSE() { return mBetasSE; }

    /**
     * @brief \~english Get a vector of trace of \f$S\f$ and \f$S'S\f$. \~chinese 获取一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量。
     * 
     * @return arma::vec \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     */
    arma::vec sHat() { return mSHat; }

    /**
     * @brief \~english Get the diagonal elements of matrix \f$Q\f$. \~chinese 获取矩阵 \f$Q\f$ 的对角线元素。
     * 
     * @return arma::vec \~english Diagonal elements of matrix \f$Q\f$ \~chinese 矩阵 \f$Q\f$ 的对角线元素
     */
    arma::vec qDiag() { return mQDiag; }

    /**
     * @brief \~english Get the hat matrix \f$S\f$. \~chinese 获取帽子矩阵 \f$S\f$。
     * 
     * @return arma::vec \~english Hat matrix \f$S\f$ \~chinese 帽子矩阵 \f$S\f$
     */
    arma::mat s() { return mS; }

public: // Algorithm
    bool isValid() override;

public: // IRegressionAnalysis
    virtual const arma::vec& dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual const arma::mat& independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual RegressionDiagnostic diagnostic() const override { return mDiagnostic; }

    virtual arma::mat predict(const arma::mat& locations) override { return arma::mat(locations.n_rows, mX.n_cols, arma::fill::zeros); }

    virtual arma::mat fit() override;

public:  // IVariableSelectable
    Status getCriterion(const std::vector<std::size_t>& variables, double& criterion) override
    {
        criterion = (this->*mIndepVarCriterionFunction)(variables);
        return mStatus;
    }

    std::vector<std::size_t> selectedVariables() override
    {
        return mSelectedIndepVars;
    }

public:  // IParallelOpenmpEnabled
    int parallelAbility() const override
    {
        return ParallelType::SerialOnly
        #ifdef ENABLE_OPENMP
        | ParallelType::OpenMP
        #endif        
        ;
    }
    
    ParallelType parallelType() const override
    {
        return mParallelType;
    }
    
    void setParallelType(const ParallelType& type) override;

    void setOmpThreadNum(const int threadNum) override
    {
        mOmpThreadNum = threadNum;
    }


public:

    /**
     * @brief \~english Calculate criterion for given bandwidths. \~chinese 获取给定带宽值对应的指标值。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthCriterion(const std::vector<BandwidthWeight*>& bandwidths)
    {
        return (this->*mBandwidthCriterionFunction)(bandwidths);
    }

protected:

    /**
     * @brief \~english Non-parallel implementation of prediction function. \~chinese 预测函数的非并行实现。
     * 
     * @param locations \~english Locations to predict \~chinese 要预测的位置
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);

    /**
     * @brief \~english Non-parallel implementation of fitting function. \~chinese 拟合函数的非并行实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betasSE \~english [out] Standard error of coefficient estimates \~chinese [出参] 回归系数估计值标准误差。
     * @param shat \~english [out] A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese [出参]  一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @param qdiag \~english [out] \~english Diagonal elements of matrix \f$Q\f$ \~chinese [出参] 矩阵 \f$Q\f$ 的对角线元素
     * @param S \~english [out] Hat matrix \f$S\f$ \~chinese [出参] 帽子矩阵 \f$S\f$
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitSerial(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qdiag, arma::mat& S);

    /**
     * @brief \~english Non-parallel implementation of calculator to get AIC criterion for given bandwidths. \~chinese 获取给定带宽值对应的AIC值的非并行实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthCriterionAICSerial(const std::vector<BandwidthWeight*>& bandwidths);

    /**
     * @brief \~english Non-parallel implementation of calculator to get CV criterion for given bandwidths. \~chinese 获取给定带宽值对应的CV值的非并行实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthCriterionCVSerial(const std::vector<BandwidthWeight*>& bandwidths);

    /**
     * @brief \~english Non-parallel implementation of calculator to get AIC criterion for given variable combination. \~chinese 获取给定变量组合对应的AIC值的非并行实现。
     * 
     * @param bandwidths \~english Given variable combination \~chinese 给定变量组合
     * @return double \~english Criterion value \~chinese 指标值
     */
    double indepVarCriterionSerial(const std::vector<std::size_t>& indepVars);

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

    /**
     * @brief \~english Multithreading implementation of fitting function. \~chinese 拟合函数的多线程实现。
     * 
     * @param x \~english Independent variables \~chinese 自变量
     * @param y \~english Dependent variables \~chinese 因变量
     * @param betasSE \~english [out] Standard error of coefficient estimates \~chinese [出参] 回归系数估计值标准误差。
     * @param shat \~english [out] A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese [出参]  一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
     * @param qdiag \~english [out] \~english Diagonal elements of matrix \f$Q\f$ \~chinese [出参] 矩阵 \f$Q\f$ 的对角线元素
     * @param S \~english [out] Hat matrix \f$S\f$ \~chinese [出参] 帽子矩阵 \f$S\f$
     * @return arma::mat \~english Coefficient estimates \~chinese 回归系数估计值
     */
    arma::mat fitOmp(const arma::mat& x, const arma::vec& y, arma::mat& betasSE, arma::vec& shat, arma::vec& qdiag, arma::mat& S);

    /**
     * @brief \~english Multithreading implementation of calculator to get AIC criterion for given bandwidths. \~chinese 获取给定带宽值对应的AIC值的多线程实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthCriterionAICOmp(const std::vector<BandwidthWeight*>& bandwidths);
    
    /**
     * @brief \~english Multithreading implementation of calculator to get CV criterion for given bandwidths. \~chinese 获取给定带宽值对应的CV值的多线程实现。
     * 
     * @param bandwidths \~english Given bandwidths \~chinese 给定带宽值
     * @return double \~english Criterion value \~chinese 指标值
     */
    double bandwidthCriterionCVOmp(const std::vector<BandwidthWeight*>& bandwidths);
    
    /**
     * @brief \~english Multithreading implementation of calculator to get AIC criterion for given variable combination. \~chinese 获取给定变量组合对应的AIC值的多线程实现。
     * 
     * @param bandwidths \~english Given variable combination \~chinese 给定变量组合
     * @return double \~english Criterion value \~chinese 指标值
     */
    double indepVarCriterionOmp(const std::vector<std::size_t>& indepVars);
#endif

private:

    /**
     * @brief \~english Get whether to store hat matrix. \~chinese 获取是否保存帽子矩阵。
     * 
     * @return true \~english Yes \~chinese 是
     * @return false \~english No \~chinese  否
     */
    bool isStoreS()
    {
        return mHasHatMatrix && (mCoords.n_rows < 8192);
    }

private:

    arma::mat mX;                       //!< \~english Dependent variables \~chinese 因变量
    arma::vec mY;                       //!< \~english Independent variables \~chinese 自变量
    std::vector<SpatialWeight> mSpatialWeights; //!< \~english Spatial weighting scheme \~chinese 空间权重配置
    arma::mat mBetas;                   //!< \~english Coefficient estimates \~chinese 回归系数估计值
    bool mHasHatMatrix = true;          //!< \~english Whether has hat matrix \~chinese 是否有帽子矩阵 
    bool mHasIntercept = true;          //!< \~english Whether has intercept \~chinese 是否包含截距 
    RegressionDiagnostic mDiagnostic = RegressionDiagnostic();   //!< \~english Diagnostic information \~chinese 诊断信息

    PredictCalculator mPredictFunction = &SDR::predictSerial;  //!< \~english Calculator to predict \~chinese 用于预测的函数
    FitCalculator mFitFunction = &SDR::fitSerial;              //!< \~english Calculator to fit \~chinese 用于拟合的函数

    bool mEnableBandwidthOptimize = false;  //!< \~english Whether bandwidth optimization is enabled \~chinese 是否进行带宽优选
    BandwidthCriterionType mBandwidthCriterionType = BandwidthCriterionType::CV;    //!< \~english Type of criterion for bandwidth optimization \~chinese 带宽优选指标类型
    BandwidthCriterionCalculator mBandwidthCriterionFunction = &SDR::bandwidthCriterionCVSerial;   //!< \~english Calculator to get criterion for given bandwidth value \~chinese 用于根据给定带宽值计算指标值的函数
    double mBandwidthOptimizeEps = 1e-6;    //!< \~english Threshold for bandwidth optimization \~chinese 带宽优选阈值
    std::size_t mBandwidthOptimizeMaxIter = 100000; //!< \~english Maximum iteration for bandwidth optimization \~chinese 带宽优选最大迭代次数
    double mBandwidthOptimizeStep = 0.01;   //!< \~english Step size for bandwidth optimization \~chinese 带宽优选步长
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。

    bool mEnableIndepVarSelect = false;     //!< \~english Whether independent variable selection is enabled \~chinese 是否优选变量
    double mIndepVarSelectThreshold = 3.0;  //!< \~english Threshold for independent variable selection \~chinese 变量优选阈值
    VariablesCriterionList mIndepVarCriterionList;  //!< \~english List of criterion values for each variable combination in independent variable selection \~chinese 变量优选过程中每种变量组合对应的指标值列表
    IndepVarCriterionCalculator mIndepVarCriterionFunction = &SDR::indepVarCriterionSerial;    //!< \~english Calculator to get criterion for given independent variable combination \~chinese 用于根据给定变量组合计算指标值的函数
    std::vector<std::size_t> mSelectedIndepVars;    //!< \~english Selected independent variable \~chinese 选中的变量组合
    std::size_t mIndepVarSelectionProgressTotal = 0; //!< \~english Total number of independent variable combination. \~chinese 自变量所有组合总数。
    std::size_t mIndepVarSelectionProgressCurrent = 0; //!< \~english Current progress of independent variable selection. \~chinese 当前自变量优选的进度。

    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Type of parallelization \~chinese 并行方法类型
    int mOmpThreadNum = 8;  //!< \~english Number of threads used in multithreading \~chinese 多线程所使用的线程数

    arma::mat mBetasSE; //!< \~english Standard error of coefficient estimates \~chinese 回归系数估计值标准误差。
    arma::vec mSHat;    //!< \~english A vector of trace of \f$S\f$ and \f$S'S\f$ \~chinese 一个包含 \f$S\f$ 和 \f$S'S\f$ 矩阵迹的向量
    arma::vec mQDiag;   //!< \~english Diagonal elements of matrix \f$Q\f$ \~chinese 矩阵 \f$Q\f$ 的对角线元素
    arma::mat mS;       //!< \~english Hat matrix \f$S\f$ \~chinese 帽子矩阵 \f$S\f$
};


class SDRBandwidthOptimizer
{
public:

    /**
     * @brief \~english Additional parameters for optimizer. \~chinese 优化器附加参数
     * 
     */
    struct Parameter
    {
        SDR* instance;     //!< \~english A SDR instance \~chinese 一个 SDR 实例
        std::vector<BandwidthWeight*>* bandwidths;  //!< \~english Bandwidths \~chinese 带宽
        arma::uword featureCount;   //!< \~english Total number of features \~chinese 要素总数
    };

    /**
     * @brief \~english Get criterion value. \~chinese 获取指标值。
     * 
     * @param bws \~english Bandwidth sizes \~chinese 带宽值
     * @param params \~english Additional parameter \~chinese 附加参数
     * @return double \~english Criterion value \~chinese 指标值
     */
    static double criterion_function(const gsl_vector* bws, void* params);

    /**
     * @brief \~english Get meta infomation of current bandwidth value and the corresponding criterion value.
     * \~chinese 获取当前带宽值和对应指标值的元信息。
     * 
     * @param weights \~english Bandwidth weight \~chinese 带宽设置
     * @return std::string \~english Information string \~chinese 信息字符串
     */
    static std::string infoBandwidthCriterion(const std::vector<BandwidthWeight*>& weights)
    {
        std::size_t number = 1;
        std::vector<std::string> labels(weights.size());
        std::transform(weights.cbegin(), weights.cend(), labels.begin(), [&number](const BandwidthWeight* bw)
        {
            return std::to_string(number++) + ":" + (bw->adaptive() ? "adaptive" : "fixed");
        });
        return std::string(GWM_LOG_TAG_BANDWIDTH_CIRTERION) + strjoin(",", labels) + ",criterion";
    }

    /**
     * @brief \~english Get infomation of current bandwidth value and the corresponding criterion value.
     * \~chinese 获取当前带宽值和对应指标值的信息。
     * 
     * @param weights \~english Bandwidth weight \~chinese 带宽设置
     * @param criterion \~english Criterion value \~chinese 指标值
     * @return std::string \~english Information string \~chinese 信息字符串
     */
    static std::string infoBandwidthCriterion(const std::vector<BandwidthWeight*>& weights, const double criterion)
    {
        std::vector<std::string> labels(weights.size());
        std::transform(weights.cbegin(), weights.cend(), labels.begin(), [](const BandwidthWeight* bw)
        {
            return std::to_string(bw->bandwidth());
        });
        return std::string(GWM_LOG_TAG_BANDWIDTH_CIRTERION) + strjoin(",", labels) + "," + std::to_string(criterion);
    }

public:

    /**
     * @brief \~english Construct a new SDRBandwidthOptimizer object. \~chinese 构造一个新的 SDRBandwidthOptimizer 对象。 
     * 
     * @param weights \~english Initial values of bandwidths \~chinese 带宽初始值
     */
    explicit SDRBandwidthOptimizer(const std::vector<BandwidthWeight*>& weights) : mBandwidths(weights) {}

    /**
     * @brief \~english Optimize bandwidth for a SDR model. \~chinese 为 SDR 模型优选带宽。
     * 
     * @param instance \~english A SDR instance \~chinese 一个 SDR 实例
     * @param featureCount \~english Total number of features \~chinese 要素总数
     * @param maxIter \~english Maximum of iteration \~chinese 最大迭代次数
     * @param eps \~english Threshold of convergence \~chinese 收敛阈值
     * @param step \~english Step size \~chinese 步长
     * @return const int \~english Optimizer status \~chinese 优化器退出状态
     */
    const int optimize(SDR* instance, arma::uword featureCount, std::size_t maxIter, double eps, double step);

private:
    std::vector<BandwidthWeight*> mBandwidths;  //!< \~english Bandwidths \~chinese 带宽
};

}

#endif  // SDR_H