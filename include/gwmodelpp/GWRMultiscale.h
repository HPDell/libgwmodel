#ifndef GWRMULTISCALE_H
#define GWRMULTISCALE_H

#include <utility>
#include <string>
#include <initializer_list>
#include <optional>
#include "SpatialMultiscaleAlgorithm.h"
#include "spatialweight/SpatialWeight.h"
#include "IRegressionAnalysis.h"
#include "IBandwidthSelectable.h"
#include "BandwidthSelector.h"
#include "IParallelizable.h"


namespace gwm
{

#define GWM_LOG_TAG_MGWR_INITIAL_BW "#initial-bandwidth "
#define GWM_LOG_TAG_MGWR_BACKFITTING "#back-fitting "

/**
 * @brief \~english Print log in `backfitting` function. \~chinese 输出 `backfitting` 函数中的日志。
 * 
 */
#define GWM_LOG_MGWR_BACKFITTING(MESSAGE) { GWM_LOG_INFO((std::string(GWM_LOG_TAG_MGWR_BACKFITTING) + (MESSAGE))); }

/**
 * \~english
 * @brief Multiscale GWR Algorithm
 * 
 * \~chinese
 * @brief 多尺度GWR算法
 */
class GWRMultiscale : public SpatialMultiscaleAlgorithm,
    public IBandwidthSelectable,
    public IRegressionAnalysis,
    public IParallelizable,
    public IParallelOpenmpEnabled,
    public IParallelCudaEnabled,
    public IParallelMpiEnabled
{
public:

    /**
     * \~english
     * @brief Type of bandwidth initilization.
     * 
     * \~chinese
     * @brief 带宽初始值类型。
     */
    enum BandwidthInitilizeType
    {
        Null,
        Initial,
        Specified
    };

    static std::unordered_map<BandwidthInitilizeType,std::string> BandwidthInitilizeTypeNameMapper; //!< \~english A mapper from bandwidth inilization types to their names. \~chinese 带宽初始值类型到其名称的映射表。

    /**
     * \~english
     * @brief Type of criterion for bandwidth selection.
     * 
     * \~chinese
     * @brief 带宽选择指标类型。
     */
    enum BandwidthSelectionCriterionType
    {
        CV,
        AIC
    };

    static std::unordered_map<BandwidthSelectionCriterionType,std::string> BandwidthSelectionCriterionTypeNameMapper;   //!< \~english A mapper from bandwidth selection criterion types to their names. \~chinese 带宽选择指标类型到其名称的映射表。

    /**
     * \~english
     * @brief Type of criterion for backfitting convergence.
     * 
     * \~chinese
     * @brief 后向迭代算法收敛指标值类型。
     */
    enum BackFittingCriterionType
    {
        CVR,
        dCVR
    };

    static std::unordered_map<BackFittingCriterionType,std::string> BackFittingCriterionTypeNameMapper; //!< \~english A mapper from backfitting convergence criterion types to their names.  \~chinese 后向迭代算法收敛指标类型到其名称的映射表。

    typedef double (GWRMultiscale::*BandwidthSizeCriterionFunction)(BandwidthWeight*);  //!< \~english Function to calculate the criterion for given bandwidth size. \~chinese 根据指定带宽大小计算对应指标值的函数。
        
    typedef arma::vec (GWRMultiscale::*FitVarFunction)(const size_t);  //!< \~english Function to fit a model for the given variable. \~chinese 根据给定变量拟合模型的函数。
        
    typedef arma::vec (GWRMultiscale::*FitVarCoreFunction)(const arma::vec&, const arma::vec&, const SpatialWeight&, arma::mat&);  //!< \~english Function to fit a model for the given variable. \~chinese 根据给定变量拟合模型的函数。
        
    typedef arma::vec (GWRMultiscale::*FitVarCoreCVFunction)(const arma::vec&, const arma::vec&, const SpatialWeight&);  //!< \~english Function to fit a model for the given variable. \~chinese 根据给定变量拟合模型的函数。
        
    typedef arma::vec (GWRMultiscale::*FitVarCoreSHatFunction)(const arma::vec&, const arma::vec&, const SpatialWeight&, arma::vec&);  //!< \~english Function to fit a model for the given variable. \~chinese 根据给定变量拟合模型的函数。
    
private:

    /**
     * \~english
     * @brief Calculate fitted values of dependent varialbe by given \f$X\f$ and \f$\beta\f$.
     * 
     * @param x Independent variables \f$X\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @return vec Fitted values of dependent varialbe.
     * 
     * \~chinese
     * @brief 根据给定的 \f$X\f$ 和 \f$\beta\f$ 计算拟合的因变量的值。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @return vec 拟合的因变量的值。
     */
    static arma::vec Fitted(const arma::mat& x, const arma::mat& betas)
    {
        return sum(betas % x, 1);
    }

    /**
     * \~english
     * @brief Calculate sum of squared residuals by given \f$X\f$, \f$y\f$, and \f$\beta\f$.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @return double Sum of squared residuals.
     * 
     * \~chinese
     * @brief 根据给定的 \f$X\f$, \f$y\f$ 和 \f$\beta\f$ 计算残差平方和。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @return double 残差平方和。
     */
    static double RSS(const arma::mat& x, const arma::mat& y, const arma::mat& betas)
    {
        arma::vec r = y - Fitted(x, betas);
        return sum(r % r);
    }

    /**
     * \~english
     * @brief Calculate AICc value according to given \f$X\f$, \f$y\f$, \f$\beta\f$, \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param betas Coefficient estimates \f$\beta\f$.
     * @param shat A vector of 2 elements: \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @return double AICc value.
     * 
     * \~chinese
     * @brief 根据给定的 \f$X\f$, \f$y\f$, \f$\beta\f$, \f$tr(S)\f$ 和 \f$tr(SS^T)\f$ 计算 AICc 值。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param betas 回归系数估计值 \f$\beta\f$。
     * @param shat 一个包含两个元素的向量，两个元素分别是 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$。
     * @return double AICc 值。
     */
    static double AICc(const arma::mat& x, const arma::mat& y, const arma::mat& betas, const arma::vec& shat)
    {
        double ss = RSS(x, y, betas), n = (double)x.n_rows;
        return n * log(ss / n) + n * log(2 * arma::datum::pi) + n * ((n + shat(0)) / (n - 2 - shat(0)));
    }

    /**
     * \~english
     * @brief Report whether to storage matrix \f$S\f$.
     * 
     * @return true Storage matrix \f$S\f$.
     * @return false Not to storage matrix \f$S\f$.
     */
    bool isStoreS()
    {
        return mHasHatMatrix && (mCoords.n_rows < 8192);
    }

    /**
     * \~english
     * @brief Calculate diagnostic information.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param shat A vector of 2 elements: \f$tr(S)\f$ and \f$tr(SS^T)\f$.
     * @param RSS Sum of squared residuals.
     * @return GwmRegressionDiagnostic Diagnostic information.
     * 
     * \~chinese
     * @brief 计算诊断信息。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param shat 一个包含两个元素的向量，两个元素分别是 \f$tr(S)\f$ 和 \f$tr(SS^T)\f$。
     * @param RSS 残差平方和。
     * @return GwmRegressionDiagnostic 诊断信息。
     */
    static RegressionDiagnostic CalcDiagnostic(const arma::mat &x, const arma::vec &y, const arma::vec &shat, double RSS);

    /**
     * \~english
     * @brief Set weighting scheme for initializing bandwidths.
     * 
     * \~chinese
     * @brief 设置初始带宽配置。
     */
    void setInitSpatialWeight(const SpatialWeight &spatialWeight)
    {
        mInitSpatialWeight = spatialWeight;
    }

public:

    /**
     * \~english
     * @brief Construct a new GWRMultiscale object.
     * 
     * \~chinese
     * @brief 构造一个新的 GWRMultiscale 对象。
     */
    GWRMultiscale() {}

    /**
     * \~english
     * @brief Construct a new GWRMultiscale object.
     * 
     * @param x Independent variables.
     * @param y Dependent variable.
     * @param coords Coordinate matrix.
     * @param spatialWeights Spatial weighting configuration.
     * 
     * \~chinese
     * @brief 构造一个新的 GWRMultiscale 对象。
     * 
     * @param x 自变量矩阵。
     * @param y 因变量。
     * @param coords 坐标矩阵。
     * @param spatialWeights 空间权重配置。
     */
    GWRMultiscale(const arma::mat& x, const arma::vec& y, const arma::mat& coords, const std::vector<SpatialWeight>& spatialWeights)
        : SpatialMultiscaleAlgorithm(coords, spatialWeights)
    {
        mX = x;
        mY = y;
        if (spatialWeights.size() > 0)
            setInitSpatialWeight(spatialWeights[0]);
    }

    /**
     * \~english
     * @brief Destroy the GWRMultiscale object.
     * 
     * \~chinese
     * @brief 销毁 GWRMultiscale 对象。
     */
    virtual ~GWRMultiscale() {}

public:

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
     * @brief Get the type of bandwidth initilization.
     * 
     * @return std::vector<BandwidthInitilizeType> The type of bandwidth initilization
     * 
     * \~chinese
     * @brief 获取带宽初始化值类型。
     * 
     * @return std::vector<BandwidthInitilizeType> 带宽初始化值类型。
     */
    const std::vector<BandwidthInitilizeType>& bandwidthInitilize() const { return GWRMultiscale::mBandwidthInitilize; }
    
    /**
     * \~english
     * @brief Set the bandwidth initilization type.
     * 
     * @param bandwidthInitilize The type of bandwidth initilization
     * 
     * \~chinese
     * @brief 设置带宽初始化值类型。
     * 
     * @param bandwidthInitilize 带宽初始化值类型。
     */
    void setBandwidthInitilize(const std::vector<BandwidthInitilizeType> &bandwidthInitilize);

    /**
     * \~english
     * @brief Get the bandwidth selection approach.
     * 
     * @return std::vector<BandwidthSelectionCriterionType> Bandwidth selection approach.
     * 
     * \~chinese
     * @brief 获取带宽选择方法。
     * 
     * @return std::vector<BandwidthSelectionCriterionType> 带宽选择方法。
     */
    const std::vector<BandwidthSelectionCriterionType>& bandwidthSelectionApproach() const { return GWRMultiscale::mBandwidthSelectionApproach; }
    
    /**
     * \~english
     * @brief Set the bandwidth selection approach.
     * 
     * @param bandwidthSelectionApproach Bandwidth selection approach.
     * 
     * \~chinese
     * @brief 设置带宽选择方法。
     * 
     * @param bandwidthSelectionApproach 带宽选择方法。
     */
    void setBandwidthSelectionApproach(const std::vector<BandwidthSelectionCriterionType> &bandwidthSelectionApproach);

    /**
     * \~english
     * @brief Get whether preditors are centered.
     * 
     * @return std::vector<bool> Whether preditors are centered.
     * 
     * \~chinese
     * @brief 获取是否中心化变量。
     * 
     * @return std::vector<bool> 是否中心化变量。
     */
    const std::vector<bool>& preditorCentered() const { return mPreditorCentered; }
    
    /**
     * \~english
     * @brief Set the Preditor Centered object
     * 
     * @param preditorCentered Whether preditors are centered.
     * 
     * \~chinese
     * @brief 设置是否中心化自变量。
     * 
     * @param preditorCentered 是否中心化自变量。
     */
    void setPreditorCentered(const std::vector<bool> &preditorCentered) { mPreditorCentered = preditorCentered; }

    /**
     * \~english
     * @brief Get the threshold of bandwidth selection.
     * 
     * @return std::vector<double> Threshold of bandwidth selection.
     * 
     * \~chinese
     * @brief 获取带宽选择阈值。
     * 
     * @return std::vector<double> 带宽选择阈值。
     */
    const std::vector<double>& bandwidthSelectThreshold() const { return mBandwidthSelectThreshold; }
    
    /**
     * \~english
     * @brief Set the threshold of bandwidth selection.
     * 
     * @param bandwidthSelectThreshold Threshold of bandwidth selection.
     * 
     * \~chinese
     * @brief 设置带宽选择阈值。
     * 
     * @param bandwidthSelectThreshold 带宽选择阈值。
     */
    void setBandwidthSelectThreshold(const std::vector<double> &bandwidthSelectThreshold) { mBandwidthSelectThreshold = bandwidthSelectThreshold; }

    /**
     * \~english
     * @brief Get whether has hat matrix \f$S\f$.
     * 
     * @return true Has hat matrix.
     * @return false Doesn't have.
     * 
     * \~chinese
     * @brief 获取是否有帽子矩阵 \f$S\f$。
     * 
     * @return true 有帽子矩阵。
     * @return false 没有帽子矩阵。
     */
    bool hasHatMatrix() const { return mHasHatMatrix; }
    
    /**
     * \~english
     * @brief Set whether has hat matrix \f$S\f$.
     * 
     * @param hasHatMatrix Whether has hat matrix \f$S\f$.
     * 
     * \~chinese
     * @brief 设置是否有帽子矩阵 \f$S\f$。
     * 
     * @param hasHatMatrix 是否有帽子矩阵 \f$S\f$。
     */
    void setHasHatMatrix(bool hasHatMatrix) { mHasHatMatrix = hasHatMatrix; }

    /**
     * \~english
     * @brief Get maximum retry times when select bandwidths.
     * 
     * @return size_t The maximum retry times when select bandwidths.
     * 
     * \~chinese
     * @brief 获取优选带宽时的最大重试次数。
     * 
     * @return size_t 优选带宽时的最大重试次数。
     */
    size_t bandwidthSelectRetryTimes() const { return (size_t)mBandwidthSelectRetryTimes; }
    
    /**
     * \~english
     * @brief Set maximum retry times when select bandwidths.
     * 
     * @param bandwidthSelectRetryTimes The maximum retry times when select bandwidths.
     * 
     * \~chinese
     * @brief 设置优选带宽时的最大重试次数。
     * 
     * @param bandwidthSelectRetryTimes 优选带宽时的最大重试次数。
     */
    void setBandwidthSelectRetryTimes(size_t bandwidthSelectRetryTimes) { mBandwidthSelectRetryTimes = (arma::uword)bandwidthSelectRetryTimes; }

    /**
     * \~english
     * @brief Get maximum iteration times.
     * 
     * @return size_t The maximum iteration times.
     * 
     * \~chinese
     * @brief 获取最大迭代次数。
     * 
     * @return size_t 最大迭代次数。
     */
    size_t maxIteration() const { return mMaxIteration; }
    
    /**
     * \~english
     * @brief Set maximum iteration times.
     * 
     * @param maxIteration The maximum iteration times.
     * 
     * \~chinese
     * @brief 设置最大迭代次数。
     * 
     * @param maxIteration 最大迭代次数。
     */
    void setMaxIteration(size_t maxIteration) { mMaxIteration = maxIteration; }

    /**
     * \~english
     * @brief Get type of backfitting convergence criterion.
     * 
     * @return BackFittingCriterionType The type of backfitting convergence criterion.
     * 
     * \~chinese
     * @brief 获取后向迭代算法收敛指标值类型。
     * 
     * @return BackFittingCriterionType 后向迭代算法收敛指标值类型。
     */
    BackFittingCriterionType criterionType() const { return mCriterionType; }
    
    /**
     * \~english
     * @brief Set type of backfitting convergence criterion.
     * 
     * @param criterionType The type of backfitting convergence criterion.
     * 
     * \~chinese
     * @brief 设置后向迭代算法收敛指标值类型。
     * 
     * @param criterionType 后向迭代算法收敛指标值类型。
     */
    void setCriterionType(const BackFittingCriterionType &criterionType) { mCriterionType = criterionType; }

    /**
     * \~english
     * @brief Get threshold of criterion.
     * 
     * @return double The threshold of criterion.
     * 
     * \~chinese
     * @brief 获取指标收敛阈值。
     * 
     * @return double 指标收敛阈值。
     */
    double criterionThreshold() const { return mCriterionThreshold; }
    
    /**
     * \~english
     * @brief Set threshold of criterion.
     * 
     * @param criterionThreshold The threshold of criterion.
     * 
     * \~chinese
     * @brief 设置指标收敛阈值。
     * 
     * @param criterionThreshold 指标收敛阈值。
     */
    void setCriterionThreshold(double criterionThreshold) { mCriterionThreshold = criterionThreshold; }

    /**
     * \~english
     * @brief Get lower bound for optimizing adaptive bandwidth.
     * 
     * @return int The lower bound for optimizing adaptive bandwidth.
     * 
     * \~chinese
     * @brief 获取优选可变带宽优选下限值。
     * 
     * @return int 优选可变带宽优选下限值。
     */
    int adaptiveLower() const { return mAdaptiveLower; }
    
    /**
     * \~english
     * @brief Set lower bound for optimizing adaptive bandwidth.
     * 
     * @param adaptiveLower The lower bound for optimizing adaptive bandwidth.
     * 
     * \~chinese
     * @brief 设置优选可变带宽优选下限值。
     * 
     * @param adaptiveLower 优选可变带宽优选下限值。
     */
    void setAdaptiveLower(int adaptiveLower) { mAdaptiveLower = adaptiveLower; }

    /**
     * \~english
     * @brief Get coefficient estimates \f$\beta\f$.
     * 
     * @return arma::mat Coefficient estimates \f$\beta\f$.
     * 
     * \~chinese
     * @brief 获取回归系数估计值 \f$\beta\f$。
     * 
     * @return arma::mat 回归系数估计值 \f$\beta\f$。
     */
    const arma::mat& betas() const { return mBetas; }
    
    /**
     * \~english
     * @brief Get criterion calculator function for optimize bandwidth size for one variable.
     * 
     * @param type The criterion type for optimize bandwidth size for one variable.
     * @return BandwidthSizeCriterionFunction The criterion calculator for optimize bandwidth size for one variable.
     * 
     * \~chinese
     * @brief 获取单个变量带宽优选的指标值计算函数。
     * 
     * @param type 单个变量带宽优选的指标值类型。
     * @return BandwidthSizeCriterionFunction 单个变量带宽优选的指标值计算函数。
     */
    BandwidthSizeCriterionFunction bandwidthSizeCriterionVar(BandwidthSelectionCriterionType type);

public:     // SpatialAlgorithm interface
    bool isValid() override;


public:     // SpatialMultiscaleAlgorithm interface
    virtual void setSpatialWeights(const std::vector<SpatialWeight> &spatialWeights) override;


public:     // IBandwidthSizeSelectable interface
    Status getCriterion(BandwidthWeight* weight, double& criterion) override
    {
        criterion = (this->*mBandwidthSizeCriterion)(weight);
        return mStatus;
    }


public:     // IRegressionAnalysis interface
    virtual const arma::vec& dependentVariable() const override { return mY; }
    virtual void setDependentVariable(const arma::vec& y) override { mY = y; }

    virtual const arma::mat& independentVariables() const override { return mX; }
    virtual void setIndependentVariables(const arma::mat& x) override { mX = x; }

    virtual bool hasIntercept() const override { return mHasIntercept; }
    virtual void setHasIntercept(const bool has) override { mHasIntercept = has; }

    virtual RegressionDiagnostic diagnostic() const override { return mDiagnostic; }

    arma::mat predict(const arma::mat& locations) override { return arma::mat(locations.n_rows, mX.n_cols, arma::fill::zeros); }

    arma::mat fit() override;

public:     // IParallelalbe interface
    int parallelAbility() const override 
    {
        return ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
            | ParallelType::OpenMP
#endif
#ifdef ENABLE_CUDA
            | ParallelType::CUDA
#endif
            ;
    }

    ParallelType parallelType() const override { return mParallelType; }

    void setParallelType(const ParallelType &type) override;


public:     // IOpenmpParallelable interface
    void setOmpThreadNum(const int threadNum) override { mOmpThreadNum = threadNum; }

public:     // IParallelCudaEnabled interface
    void setGPUId(const int gpuId) override { mGpuId = gpuId; }
    void setGroupSize(const double size) override { mGroupLength = size; }
    int workerId() override { return mWorkerId; }
    void setWorkerId(int id) override { mWorkerId = id; };
    void setWorkerNum(int size) override { mWorkerNum = size; };

protected:

    /**
     * \~english
     * @brief Get the \f$i\f$-th bandwidth.
     * 
     * @param i The index of bandwidth.
     * @return BandwidthWeight* The \f$i\f$-th bandwidth.
     * 
     * \~chinese
     * @brief 获取第 \f$i\f$ 个带宽。
     * 
     * @param i 带宽索引值。
     * @return BandwidthWeight* 第 \f$i\f$ 个带宽。
     */
    BandwidthWeight* bandwidth(size_t i)
    {
        return mSpatialWeights[i].weight<BandwidthWeight>();
    }

    /**
     * \~english
     * @brief The backfitting algorithm.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @return arma::mat Coefficient estimates \f$\beta\f$.
     * 
     * \~chinese
     * @brief 后向迭代算法。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @return arma::mat 回归系数估计值 \f$\beta\f$。
     */
    arma::mat backfitting(const arma::mat& betas0);

    arma::vec fitVarBase(const size_t var);
    
    /**
     * \~english
     * @brief The serial implementation of CV criterion calculator for given bandwidth size and one variable.
     * 
     * @param bandwidthWeight Badwidth weight.
     * @return double CV criterion value.
     * 
     * \~chinese
     * @brief 为指定带宽值和某个变量计算CV指标值函数的非并行实现。
     * 
     * @param bandwidthWeight 带宽值。
     * @return double CV指标值。
     */
    double bandwidthSizeCriterionVarCVBase(BandwidthWeight* bandwidthWeight);
    
    /**
     * \~english
     * @brief The serial implementation of AIC criterion calculator for given bandwidth size and one variable.
     * 
     * @param bandwidthWeight Badwidth weight.
     * @return double AIC criterion value.
     * 
     * \~chinese
     * @brief 为指定带宽值和某个变量计算AIC指标值函数的非并行实现。
     * 
     * @param bandwidthWeight 带宽值。
     * @return double AIC指标值。
     */
    double bandwidthSizeCriterionVarAICBase(BandwidthWeight* bandwidthWeight);

    /**
     * \~english
     * @brief The serial implementation of fit function for one variable.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param var The index of this variable.
     * @param S The hat matrix \f$S\f$.
     * @return arma::vec The coefficient estimates corresponding to this variable.
     * 
     * \~chinese
     * @brief 拟合单个变量的非并行实现。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param var 当前变量的索引值。
     * @param S 帽子矩阵 \f$S\f$
     * @return arma::vec 该变量对应的回归系数估计值。
     */
    arma::vec fitVarCoreSerial(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw, arma::mat& S);

    arma::vec fitVarCoreCVSerial(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw);

    arma::vec fitVarCoreSHatSerial(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw, arma::vec& shat);

#ifdef ENABLE_OPENMP
    arma::vec fitVarCoreOmp(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw, arma::mat& S);

    arma::vec fitVarCoreCVOmp(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw);

    arma::vec fitVarCoreSHatOmp(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw, arma::vec& shat);
#endif

#ifdef ENABLE_CUDA
    
    /**
     * \~english
     * @brief The CUDA implementation of fit function for one variable.
     * 
     * @param x Independent variables \f$X\f$.
     * @param y Dependent variable \f$y\f$.
     * @param var The index of this variable.
     * @param S The hat matrix \f$S\f$.
     * @return arma::vec The coefficient estimates corresponding to this variable.
     * 
     * \~chinese
     * @brief 拟合单个变量的CUDA实现。
     * 
     * @param x 自变量矩阵 \f$X\f$。
     * @param y 因变量 \f$y\f$。
     * @param var 当前变量的索引值。
     * @param S 帽子矩阵 \f$S\f$
     * @return arma::vec 该变量对应的回归系数估计值。
     */
    arma::vec fitVarCoreCuda(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw, arma::mat& S);
    
    arma::vec fitVarCoreCVCuda(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw);
    
    arma::vec fitVarCoreSHatCuda(const arma::vec& x, const arma::vec& y, const SpatialWeight& sw, arma::vec& shat);

#endif // ENABLE_CUDA

    /**
     * \~english
     * @brief Create a Initial Distance Parameter object
     * 
     * \~chinese
     * @brief 创建初始距离参数对象。
     */
    void createInitialDistanceParameter();

private:
    FitVarFunction mFitVar = &GWRMultiscale::fitVarBase;  //!< \~english Calculator to fit a model for one variable. \~chinese 为单一变量拟合模型的函数。
    FitVarCoreFunction mFitVarCore = &GWRMultiscale::fitVarCoreSerial;  //!< \~english Calculator to fit a model for one variable. \~chinese 为单一变量拟合模型的函数。
    FitVarCoreCVFunction mFitVarCoreCV = &GWRMultiscale::fitVarCoreCVSerial;  //!< \~english Calculator to fit a model for one variable. \~chinese 为单一变量拟合模型的函数。
    FitVarCoreSHatFunction mFitVarCoreSHat = &GWRMultiscale::fitVarCoreSHatSerial;  //!< \~english Calculator to fit a model for one variable. \~chinese 为单一变量拟合模型的函数。

    SpatialWeight mInitSpatialWeight;   //!< \~english Spatial weighting sheme for initializing bandwidth. \~chinese 计算初始带宽值时所用的空间权重配置。
    BandwidthSizeCriterionFunction mBandwidthSizeCriterion = &GWRMultiscale::bandwidthSizeCriterionVarCVBase; //!< \~english The criterion calculator for given bandwidth size. \~chinese 根据指定带宽值计算指标值的函数。
    size_t mBandwidthSelectionCurrentIndex = 0; //!< \~english The index of variable which currently the algorithm select bandwidth for. \~chinese 当前正在选带宽的变量索引值。
    double mBandwidthLastCriterion = DBL_MAX;   //!< \~english Last criterion for bandwidth selection. \~chinese 上一次带宽优选的有效指标值。
    std::optional<double> mGoldenUpperBounds;
    std::optional<double> mGoldenLowerBounds;
    std::vector<double> mMaxDistances;
    std::vector<double> mMinDistances;

    std::vector<BandwidthInitilizeType> mBandwidthInitilize;    //!< \~english Type of bandwidth initilization values. \~chinese 带宽初始值类型。
    std::vector<BandwidthSelectionCriterionType> mBandwidthSelectionApproach;   //!< \~english Type of bandwidth selection approach. \~chinese 带宽选择方法类型。
    std::vector<bool> mPreditorCentered;    //!< \~english Whether each independent variable is centered. \~chinese 每个变量是否被中心化。
    std::vector<double> mBandwidthSelectThreshold;  //!< \~english Threshold of bandwidth selection. \~chinese 带宽选择阈值。
    arma::uword mBandwidthSelectRetryTimes = 5; //!< \~english The maximum retry times when select bandwidths. \~chinese 优选带宽时的最大重试次数。
    size_t mMaxIteration = 500; //!< \~english The maximum iteration times. \~chinese 最大迭代次数。
    BackFittingCriterionType mCriterionType = BackFittingCriterionType::dCVR;   //!< \~english The type of backfitting convergence criterion. \~chinese 后向迭代算法收敛指标值类型。
    double mCriterionThreshold = 1e-6;  //!< \~english The threshold of criterion. \~chinese 指标收敛阈值。
    int mAdaptiveLower = 10;    //!< \~english The lower bound for optimizing adaptive bandwidth. \~chinese 优选可变带宽优选下限值。

    bool mHasHatMatrix = true;  //!< \~english  \~chinese

    arma::mat mX;   //!< \~english Independent variables \f$X\f$. \~chinese 自变量矩阵 \f$X\f$。
    arma::vec mY;   //!< \~english endent variable \f$y\f$. \~chinese 因变量 \f$y\f$。
    arma::mat mBetas;   //!< \~english Coefficient estimates \f$\beta\f$. \~chinese 回归系数估计值 \f$\beta\f$。
    arma::mat mBetasSE; //!< \~english Standard errors of coefficient estimates. \~chinese 回归系数估计值标准差。
    arma::mat mBetasTV; //!< \~english T-test value of coefficient estimates. \~chinese 回归系数t检验值。
    bool mHasIntercept = true;  //!< \~english Whether has hat matrix \f$S\f$. \~chinese 是否有帽子矩阵 \f$S\f$。

    arma::mat mS0;  //!< \~english  \~chinese
    arma::cube mSArray; //!< \~english  \~chinese
    arma::cube mC;  //!< \~english  \~chinese
    arma::mat mX0;  //!< \~english  \~chinese
    arma::vec mY0;  //!< \~english  \~chinese
    arma::vec mXi;  //!< \~english  \~chinese
    arma::vec mYi;  //!< \~english  \~chinese

    double mRSS0;   //!< \~english  \~chinese

    RegressionDiagnostic mDiagnostic;   //!< \~english Diagnostic information. \~chinese 诊断信息。

    ParallelType mParallelType = ParallelType::SerialOnly;  //!< \~english Parallel type of this algorithm. \~chinese 当前算法的并行类型。
    int mOmpThreadNum = 8;  //!< \~english Number of threads. \~chinese 并行线程数。
    size_t mGroupLength = 64;   //!< \~english Size of a group computing together. \~chinese 同时计算的一组的大小。
    int mGpuId = 0; //!< \~english The ID of selected GPU. \~chinese 选择的 GPU 的 ID。
    int mWorkerId = 0;
    int mWorkerNum = 1;
    // arma::uword mWorkRangeSize = 0;
    std::optional<std::pair<arma::uword, arma::uword>> mWorkRange;

public:
    static int treeChildCount;  //!< \~english  \~chinese
};

}

#endif // GWRMULTISCALE_H
