#ifndef GWRBASIC_H
#define GWRBASIC_H

#include <utility>
#include <string>
#include <initializer_list>
#include <optional>
#include "GWRBase.h"
#include "RegressionDiagnostic.h"
#include "IBandwidthSelectable.h"
#include "IVarialbeSelectable.h"
#include "IParallelizable.h"

namespace gwm
{

/**
 * \~english
 * @brief Basic implementation of geographically weighted regression.
 * This algorithm can auto select bandwidth and variables.
 * This algorithm can speed-up by OpenMP.
 * 
 * \~chinese
 * @brief 基础地理加权回归算法的实现。
 * 该算法可以自动选带宽和变量。
 * 该算法可以通过 OpenMP 加速。
 * 
 */
class GWRBasic : public GWRBase, public IBandwidthSelectable, public IVarialbeSelectable, public IParallelizable, public IParallelOpenmpEnabled, public IParallelCudaEnabled, public IParallelMpiEnabled
{
public:

    /**
     * \~english
     * @brief Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 用于带宽优选的指标类型。
     * 
     */
    enum BandwidthSelectionCriterionType
    {
        AIC,    //!< AIC
        CV      //!< CV
    };

    static std::unordered_map<BandwidthSelectionCriterionType, std::string> BandwidthSelectionCriterionTypeNameMapper;
    
    typedef arma::mat (GWRBasic::*PredictCalculator)(const arma::mat&, const arma::mat&, const arma::vec&);                             //!< \~english Predict function declaration. \~chinese 预测函数声明。
    typedef arma::mat (GWRBasic::*FitCalculator)();   //!< \~english Fit function declaration. \~chinese 拟合函数声明。
    typedef arma::mat (GWRBasic::*FitCoreCalculator)(const arma::mat&, const arma::vec&, const SpatialWeight&);   //!< \~english Fit function declaration. \~chinese 拟合函数声明。
    typedef arma::mat (GWRBasic::*FitCoreSHatCalculator)(const arma::mat&, const arma::vec&, const SpatialWeight&, arma::vec&);   //!< \~english Fit function declaration. \~chinese 拟合函数声明。
    typedef arma::mat (GWRBasic::*FitCoreCVCalculator)(const arma::mat&, const arma::vec&, const SpatialWeight&);   //!< \~english Fit function declaration. \~chinese 拟合函数声明。

    typedef double (GWRBasic::*BandwidthSelectionCriterionCalculator)(BandwidthWeight*);        //!< \~english Declaration of criterion calculator for bandwidth selection. \~chinese 带宽优选指标计算函数声明。
    typedef double (GWRBasic::*IndepVarsSelectCriterionCalculator)(const std::vector<std::size_t>&); //!< \~english Declaration of criterion calculator for variable selection. \~chinese 变量优选指标计算函数声明。

private:

    /**
     * \~english
     * @brief Calculate diagnostic information.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @param shat A vector of 2 elements: \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @return GwmRegressionDiagnostic Diagnostic information.
     * 
     * \~chinese
     * @brief 计算诊断信息。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @param shat 一个包含两个元素的向量，两个元素分别是 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$。
     * @return GwmRegressionDiagnostic 诊断信息。
     * 
     */
    static RegressionDiagnostic CalcDiagnostic(const arma::mat& x, const arma::vec& y, const arma::mat& betas, const arma::vec& shat);

public:
    
    /**
     * \~english
     * @brief Construct a new CGwmGWRBasic object.
     * 
     * \~chinese
     * @brief 构造 CGwmGWRBasic 对象。
     * 
     */
    GWRBasic() {}

    /**
     * \~english
     * @brief Construct a new CGwmGWRBasic object.
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param coords Coordinate matrix.
     * @param spatialWeight Spatial weighting configuration.
     * @param hasHatMatrix Whether has hat-matrix.
     * @param hasIntercept Whether has intercept.
     * 
     * \~chinese
     * @brief 构造 CGwmGWRBasic 对象。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param coords 坐标矩阵。
     * @param spatialWeight 空间权重配置。
     * @param hasHatMatrix 是否计算帽子矩阵。
     * @param hasIntercept 是否有截距。
     */
    GWRBasic(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const SpatialWeight& spatialWeight, bool hasHatMatrix = true, bool hasIntercept = true)
        : GWRBase(x, y, spatialWeight, coords)
    {
        mHasHatMatrix = hasHatMatrix;
        mHasIntercept = hasIntercept;
    }
    
    /**
     * \~english
     * @brief Destroy the CGwmGWRBasic object.
     * 
     * \~chinese
     * @brief 析构 CGwmGWRBasic 对象。
     * 
     */
    ~GWRBasic() {}

public:

    /**
     * \~english
     * @brief Get whether auto select bandwidth.
     * 
     * @return true if auto select bandwidth.
     * @return false if not auto select bandwidth.
     * 
     * \~chinese
     * @brief 获取是否自动优选带宽。
     * 
     * @return true 如果自动优选带宽。
     * @return false 如果不自动优选带宽。
     * 
     */
    bool isAutoselectBandwidth() const { return mIsAutoselectBandwidth; }

    /**
     * \~english
     * @brief Set whether auto select bandwidth.
     * 
     * @param isAutoSelect true if auto select bandwidth, otherwise false.
     * 
     * \~chinese
     * @brief 设置是否自动优选带宽。
     * 
     * @param isAutoSelect true 如果要自动优选带宽，否则 false。
     */
    void setIsAutoselectBandwidth(bool isAutoSelect) { mIsAutoselectBandwidth = isAutoSelect; }

    /**
     * \~english
     * @brief Get type of criterion for bandwidth selection.
     * 
     * @return BandwidthSelectionCriterionType Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 获取带宽自动优选指标值类型。
     * 
     * @return BandwidthSelectionCriterionType 带宽自动优选指标值类型。
     */
    BandwidthSelectionCriterionType bandwidthSelectionCriterion() const { return mBandwidthSelectionCriterion; }

    /**
     * \~english
     * @brief Set type of criterion for bandwidth selection.
     * 
     * @param criterion Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 设置带宽自动优选指标值类型。
     * 
     * @param criterion 带宽自动优选指标值类型。
     */
    void setBandwidthSelectionCriterion(const BandwidthSelectionCriterionType& criterion);

    /**
     * @brief \~english Set the upper bounds of golden selection. \~chinese 设置 Golden selection 算法的上界。
     * 
     * @param value \~english \~chinese
     */
    void setGoldenUpperBounds(double value) { mGoldenUpperBounds = value; }

    /**
     * @brief \~english Set the lower bounds of golden selection. \~chinese 设置 Golden selection 算法的下界。
     * 
     * @param value \~english \~chinese
     */
    void setGoldenLowerBounds(double value) { mGoldenLowerBounds = value; }

    /**
     * \~english
     * @brief Get whether auto select variables.
     * 
     * @return true if auto select variables.
     * @return false if not auto select variables.
     * 
     * \~chinese
     * @brief 获取是否自动优选变量。
     * 
     * @return true 如果自动优选变量。
     * @return false 如果不自动优选变量。
     */
    bool isAutoselectIndepVars() const { return mIsAutoselectIndepVars; }

    /**
     * \~english
     * @brief Set whether auto select variables.
     * 
     * @param isAutoSelect true if auto select variables, otherwise false.
     * 
     * \~chinese
     * @brief 设置是否自动优选变量。
     * 
     * @param isAutoSelect true 如果要自动优选变量，否则 false。
     */
    void setIsAutoselectIndepVars(bool isAutoSelect) { mIsAutoselectIndepVars = isAutoSelect; }

    /**
     * \~english
     * @brief Get threshold for variable selection.
     * 
     * @return double Threshold for variable selection.
     * 
     * \~chinese
     * @brief 获取变量优选指标类型。
     * 
     * @return double 变量优选指标类型。
     */
    double indepVarSelectionThreshold() const { return mIndepVarSelectionThreshold; }

    /**
     * \~english
     * @brief Set threshold for variable selection.
     * 
     * @param threshold Threshold for variable selection.
     * This value dependends on the size of samples.
     * Larger samples, larger threshold.
     * 
     * \~chinese
     * @brief 设置变量优选指标类型。
     * 
     * @param threshold 变量优选指标类型。
     * 该值的大小取决于样本的数量。
     * 样本数量越多，值越大。
     */
    void setIndepVarSelectionThreshold(double threshold) { mIndepVarSelectionThreshold = threshold; }
    
    /**
     * \~english
     * @brief Get criterion list for variable selection.
     * 
     * @return VariablesCriterionList Criterion list for variable selection.
     * 
     * \~chinese
     * @brief 获取变量优选过程的指标值列表。
     * 
     * @return VariablesCriterionList 变量优选过程的指标值列表。
     */
    const VariablesCriterionList& indepVarsSelectionCriterionList() const { return mIndepVarsSelectionCriterionList; }

    /**
     * \~english
     * @brief Get criterion list for bandwidth selection.
     * 
     * @return BandwidthCriterionList Criterion list for bandwidth selection.
     * 
     * \~chinese
     * @brief 获取带宽优选过程的指标值列表。
     * 
     * @return BandwidthCriterionList 带宽优选过程的指标值列表。
     */
    const BandwidthCriterionList& bandwidthSelectionCriterionList() const { return mBandwidthSelectionCriterionList; }

    /**
     * \~english
     * @brief Get whether has hat-matrix.
     * 
     * @return true if has hat-matrix.
     * @return false if does not have hat-matrix.
     * 
     * \~chinese
     * @brief 获取是否计算帽子矩阵。
     * 
     * @return true 如果计算帽子矩阵。
     * @return false 如果不计算帽子矩阵。
     */
    bool hasHatMatrix() const { return mHasHatMatrix; }

    /**
     * \~english
     * @brief Set the Has Hat-Matrix object
     * 
     * @param has true if has hat-matrix, otherwise false.
     * 
     * \~chinese
     * @brief 设置是否计算帽子矩阵。
     * 
     * @param has true 如果计算帽子矩阵，否则 false。
     */
    void setHasHatMatrix(const bool has) { mHasHatMatrix = has; }

    /**
     * \~english
     * @brief Get standard errors of coefficient estimates.
     * 
     * @return arma::mat Standard errors of coefficient estimates.
     * 
     * \~chinese
     * @brief 获取回归系数估计值的标准差。
     * 
     * @return arma::mat 回归系数估计值的标准差。
     */
    const arma::mat& betasSE() { return mBetasSE; }

    /**
     * \~english
     * @brief Get a vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * 
     * @return arma::vec A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * 
     * \~chinese
     * @brief 获取一个由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * 
     * @return arma::vec 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     */
    const arma::vec& sHat() { return mSHat; }

    /**
     * \~english
     * @brief Get the diagonal elements of matrix \f$Q\f$.
     * 
     * @return arma::vec The diagonal elements of matrix \f$Q\f$.
     * 
     * \~chinese
     * @brief 获取矩阵 \f$Q\f$ 的对角线元素。
     * 
     * @return arma::vec 矩阵 \f$Q\f$ 的对角线元素。
     */
    const arma::vec& qDiag() { return mQDiag; }

    /**
     * \~english
     * @brief Get the hat-matrix \f$S\f$.
     * 
     * @return arma::mat The hat-matrix \f$S\f$.
     * 
     * \~chinese
     * @brief 获取帽子矩阵 \f$S\f$。
     * 
     * @return arma::mat 帽子矩阵 \f$S\f$。
     */
    const arma::mat& s() { return mS; }

    const arma::cube& c() { return mC; }

    /**
     * \~english
     * @brief Whether to store hat-matrix \f$S\f$.
     * 
     * @return true if store hat-matrix.
     * @return false if not to store hat-matrix.
     * 
     * \~chinese
     * @brief 是否保存帽子矩阵 \f$S\f$.
     * 
     * @return true 如果保存帽子矩阵。
     * @return false 如果不保存帽子矩阵。
     * 
     */
    bool isStoreS() { return mStoreS ? true : (mHasHatMatrix && (mCoords.n_rows < 8192)); }

    bool isStoreC() { return mStoreC; }

    void setStoreS(bool flag) { mStoreS = flag; }

    void setStoreC(bool flag) { mStoreC = flag; }

public:     // Implement Algorithm
    bool isValid() override;

public:     // Implement IRegressionAnalysis
    arma::mat predict(const arma::mat& locations) override;

    arma::mat fit() override;

public:     // Implement IVariableSelectable
    Status getCriterion(const std::vector<size_t>& variables, double& criterion) override
    {
        criterion = (this->*mIndepVarsSelectionCriterionFunction)(variables);
        return mStatus;
    }

    std::vector<std::size_t> selectedVariables() override
    {
        return mSelectedIndepVars;
    }
    
    /**
     * \~english
     * @brief Get AIC value with given variables for variable optimization (serial implementation).
     * 
     * @param indepVars Given variables
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的变量计算变量优选的AIC值（串行实现）。
     * 
     * @param indepVars 指定的变量。
     * @return double 变量优选的指标值。
     */
    double indepVarsSelectionCriterion(const std::vector<std::size_t>& indepVars);


public:     // Implement IBandwidthSelectable
    Status getCriterion(BandwidthWeight* weight, double& criterion) override
    {
        criterion = (this->*mBandwidthSelectionCriterionFunction)(weight);
        return mStatus;
    }
    
    /**
     * \~english
     * @brief Get CV value with given bandwidth for bandwidth optimization (serial implementation).
     * 
     * @param bandwidthWeight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的CV值（串行实现）。
     * 
     * @param bandwidthWeight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionCV(BandwidthWeight* bandwidthWeight);
    
    /**
     * \~english
     * @brief Get AIC value with given bandwidth for bandwidth optimization (serial implementation).
     * 
     * @param bandwidthWeight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的AIC值（串行实现）。
     * 
     * @param bandwidthWeight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    double bandwidthSizeCriterionAIC(BandwidthWeight* bandwidthWeight);


private:
    
    /**
     * \~english 
     * @brief Predict coefficients on specified locations (serial implementation).
     * 
     * @param locations Locations where to predict coefficients.
     * @param x Independent variables.
     * @param y Dependent variable.
     * @return mat Predicted coefficients.
     * 
     * \~chinese 
     * @brief 在指定位置处进行回归系数预测（单线程实现）。
     * 
     * @param locations 指定位置。
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @return mat 回归系数预测值。
     */
    arma::mat predictSerial(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    
    /**
     * \~english
     * @brief Fit coefficients (serial implementation).
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param betasSE [out] Standard errors of coefficient estimates.
     * @param shat [out] A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @param qDiag [out] The diagonal elements of matrix \f$Q\f$.
     * @param S [out] The hat-matrix \f$S\f$.
     * @return mat Coefficient estimates.
     * 
     * \~chinese
     * @brief 回归系数估计值（串行实现）。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param betasSE [out] 回归系数估计值的标准差。
     * @param shat [out] 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * @param qDiag [out] 矩阵 \f$Q\f$ 的对角线元素。
     * @param S [out] 帽子矩阵 \f$S\f$。
     * @return mat 回归系数估计值
     */
    arma::mat fitBase();


private:

    arma::mat fitCoreSerial(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw);

    arma::mat fitCoreSHatSerial(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw, arma::vec& shat);

    arma::mat fitCoreCVSerial(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw);

#ifdef ENABLE_OPENMP

    /**
     * \~english 
     * @brief Predict coefficients on specified locations (OpenMP implementation).
     * 
     * @param locations Locations where to predict coefficients.
     * @param x Independent variables.
     * @param y Dependent variable.
     * @return mat Predicted coefficients.
     * 
     * \~chinese 
     * @brief 在指定位置处进行回归系数预测（OpenMP 实现）。
     * 
     * @param locations 指定位置。
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @return mat 回归系数预测值。
     */
    arma::mat predictOmp(const arma::mat& locations, const arma::mat& x, const arma::vec& y);
    
    /**
     * \~english
     * @brief Fit coefficients (OpenMP implementation).
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param betasSE [out] Standard errors of coefficient estimates.
     * @param shat [out] A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @param qDiag [out] The diagonal elements of matrix \f$Q\f$.
     * @param S [out] The hat-matrix \f$S\f$.
     * @return mat Coefficient estimates.
     * 
     * \~chinese
     * @brief 回归系数估计值（OpenMP 实现）。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param betasSE [out] 回归系数估计值的标准差。
     * @param shat [out] 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * @param qDiag [out] 矩阵 \f$Q\f$ 的对角线元素。
     * @param S [out] 帽子矩阵 \f$S\f$。
     * @return mat 回归系数估计值
     */
    arma::mat fitCoreOmp(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);

    /**
     * \~english
     * @brief Get CV value with given bandwidth for bandwidth optimization (OpenMP implementation).
     * 
     * @param bandwidthWeight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的CV值（OpenMP 实现）。
     * 
     * @param bandwidthWeight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    arma::mat fitCoreCVOmp(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw);

    /**
     * \~english
     * @brief Get AIC value with given variables for variable optimization (OpenMP implementation).
     * 
     * @param indepVars Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的变量计算变量优选的AIC值（OpenMP 实现）。
     * 
     * @param indepVars 指定的变量。
     * @return double 变量优选的指标值。
     */
    arma::mat fitCoreSHatOmp(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw, arma::vec& shat);

#endif

#ifdef ENABLE_CUDA

    /**
     * \~english 
     * @brief Predict coefficients on specified locations (CUDA implementation).
     * 
     * @param locations Locations where to predict coefficients.
     * @param x Independent variables.
     * @param y Dependent variable.
     * @return mat Predicted coefficients.
     * 
     * \~chinese 
     * @brief 在指定位置处进行回归系数预测（CUDA实现）。
     * 
     * @param locations 指定位置。
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @return mat 回归系数预测值。
     */
    arma::mat fitCoreCuda(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw, arma::mat& betasSE, arma::vec& shat, arma::vec& qDiag, arma::mat& S);

    /**
     * \~english
     * @brief Fit coefficients (CUDA implementation).
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param betasSE [out] Standard errors of coefficient estimates.
     * @param shat [out] A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @param qDiag [out] The diagonal elements of matrix \f$Q\f$.
     * @param S [out] The hat-matrix \f$S\f$.
     * @return mat Coefficient estimates.
     * 
     * \~chinese
     * @brief 回归系数估计值（CUDA实现）。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param betasSE [out] 回归系数估计值的标准差。
     * @param shat [out] 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
     * @param qDiag [out] 矩阵 \f$Q\f$ 的对角线元素。
     * @param S [out] 帽子矩阵 \f$S\f$。
     * @return mat 回归系数估计值
     */
    arma::mat predictCuda(const arma::mat& locations, const arma::mat& x, const arma::vec& y);

    /**
     * \~english
     * @brief Get CV value with given bandwidth for bandwidth optimization (CUDA implementation).
     * 
     * @param bandwidthWeight Given bandwidth
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的带宽计算带宽优选的CV值（CUDA实现）。
     * 
     * @param bandwidthWeight 指定的带宽。
     * @return double 带宽优选的指标值。
     */
    arma::mat fitCoreCVCuda(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw);

    /**
     * \~english
     * @brief Get AIC value with given variables for variable optimization (CUDA implementation).
     * 
     * @param indepVars Given variables
     * @return double Criterion value
     * 
     * \~chinese
     * @brief 根据指定的变量计算变量优选的AIC值（CUDA实现）。
     * 
     * @param indepVars 指定的变量。
     * @return double 变量优选的指标值。
     */
    arma::mat fitCoreSHatCuda(const arma::mat& x, const arma::vec& y, const SpatialWeight& sw, arma::vec& shat);

#endif

#ifdef ENABLE_MPI
    double indepVarsSelectionCriterionMpi(const std::vector<std::size_t>& indepVars);
    double bandwidthSizeCriterionCVMpi(BandwidthWeight* bandwidthWeight);
    double bandwidthSizeCriterionAICMpi(BandwidthWeight* bandwidthWeight);
    arma::mat fitMpi();
#endif // ENABLE_MPI

public:     // Implement IParallelizable
    int parallelAbility() const override
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
            | ParallelType::CUDA
#endif // ENABLE_CUDA
#ifdef ENABLE_MPI
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
            | ParallelType::CUDA
#endif // ENABLE_CUDA
#endif // ENABLE_MPI
        ;
    }

    ParallelType parallelType() const override { return mParallelType; }

    void setParallelType(const ParallelType& type) override;

public:     // Implement IGwmParallelOpenmpEnabled
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }
    void setGPUId(const int gpuId) override { mGpuId = gpuId; };
    void setGroupSize(const double size) override { mGroupLength = size; };
    int workerId() override { return mWorkerId; }
    void setWorkerId(int id) override { mWorkerId = id; };
    void setWorkerNum(int size) override { mWorkerNum = size; };

protected:

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

protected:
    bool mHasHatMatrix = true;  //!< \~english Whether has hat-matrix. \~chinese 是否具有帽子矩阵。
    bool mHasFTest = false;  //!< @todo \~english Whether has F-test \~chinese 是否具有F检验。
    bool mHasPredict = false;  //!< @deprecated \~english Whether has variables to predict dependent variable. \~chinese 是否有预测位置处的变量。
    
    bool mIsAutoselectIndepVars = false;    //!< \~english Whether to auto select variables. \~chinese 是否自动优选变量。
    double mIndepVarSelectionThreshold = 3.0;   //!< \~english The threshold for variable selection. \~chinese 变量优选的阈值。
    IndepVarsSelectCriterionCalculator mIndepVarsSelectionCriterionFunction = &GWRBasic::indepVarsSelectionCriterion; //!< \~english Criterion calculator for variable selection. \~chinese 变量优选的指标计算函数。
    VariablesCriterionList mIndepVarsSelectionCriterionList;    //!< \~english Criterion list of each variable combination. \~chinese 每种变量组合对应的指标值。
    std::vector<std::size_t> mSelectedIndepVars;    //!< \~english Selected variables. \~chinese 优选得到的变量。
    std::size_t mIndepVarSelectionProgressTotal = 0; //!< \~english Total number of independent variable combination. \~chinese 自变量所有组合总数。
    std::size_t mIndepVarSelectionProgressCurrent = 0; //!< \~english Current progress of independent variable selection. \~chinese 当前自变量优选的进度。

    bool mIsAutoselectBandwidth = false;    //!< \~english Whether to auto select bandwidth. \~chinese 是否自动优选带宽。
    BandwidthSelectionCriterionType mBandwidthSelectionCriterion = BandwidthSelectionCriterionType::AIC;    //!< \~english Type criterion for bandwidth selection. \~chinese 带宽优选的指标值类型。
    BandwidthSelectionCriterionCalculator mBandwidthSelectionCriterionFunction = &GWRBasic::bandwidthSizeCriterionCV; //!< \~english Criterion calculator for bandwidth selection. \~chinese 带宽优选的指标计算函数。
    BandwidthCriterionList mBandwidthSelectionCriterionList;    //!< \~english Criterion list of each bandwidth. \~chinese 每种带宽组合对应的指标值。
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。
    std::optional<double> mGoldenUpperBounds;
    std::optional<double> mGoldenLowerBounds;

    PredictCalculator mPredictFunction = &GWRBasic::predictSerial;  //!< \~english Implementation of predict function. \~chinese 预测的具体实现函数。
    FitCalculator mFitFunction = &GWRBasic::fitBase;  //!< \~english Implementation of fit function. \~chinese 拟合的具体实现函数。
    FitCoreCalculator mFitCoreFunction = &GWRBasic::fitCoreSerial;  //!< \~english Implementation of fit function. \~chinese 拟合的具体实现函数。
    FitCoreSHatCalculator mFitCoreSHatFunction = &GWRBasic::fitCoreSHatSerial;  //!< \~english Implementation of fit function. \~chinese 拟合的具体实现函数。
    FitCoreCVCalculator mFitCoreCVFunction = &GWRBasic::fitCoreCVSerial;  //!< \~english Implementation of fit function. \~chinese 拟合的具体实现函数。

    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Type of parallel method. \~chinese 并行方法类型。
    int mOmpThreadNum = 8;  //!< \~english Number of threads to create. \~chinese 并行计算创建的线程数。
    size_t mGroupLength = 64;   //!< \~english Size of a group computing together. \~chinese 同时计算的一组的大小。
    int mGpuId = 0; //!< \~english The ID of selected GPU. \~chinese 选择的 GPU 的 ID。
    int mWorkerId = 0;
    int mWorkerNum = 1;
    arma::uword mWorkRangeSize = 0;
    std::optional<std::pair<arma::uword, arma::uword>> mWorkRange;

    arma::mat mBetasSE;  //!< \~english Standard errors of coefficient estimates. \~chinese 回归系数估计值的标准差。
    arma::vec mSHat;  //!< \~english A vector of \f$tr(S)\f$ and \f$tr(SS^T)\f$. \~chinese 由 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 组成的向量。
    arma::vec mQDiag;  //!< \~english The diagonal elements of matrix \f$Q\f$. \~chinese 矩阵 \f$Q\f$ 的对角线元素。
    arma::mat mS;  //!< \~english The hat-matrix \f$S\f$. \~chinese 帽子矩阵 \f$S\f$。
    arma::cube mC;//!< \~english All \f$S\f$ matrices. \~chinese 所有 \f$C\f$ 矩阵。
    bool mStoreS = false; //!< \~english Whether to save S \~chinese 是否保存 S 矩阵
    bool mStoreC = false; //!< \~english Whether to save C \~chinese 是否保存 C 矩阵
};

}

#endif  // GWRBASIC_H