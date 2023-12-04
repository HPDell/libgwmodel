#ifndef SPATIALWEIGHT_H
#define SPATIALWEIGHT_H

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "gwmodelpp/spatialweight/cuda/ISpatialCudaEnabled.h"
#endif 

#include "Weight.h"
#include "Distance.h"

#include "BandwidthWeight.h"

#include "CRSDistance.h"
#include "MinkwoskiDistance.h"
#include "DMatDistance.h"
#include "OneDimDistance.h"
#include "CRSSTDistance.h"

namespace gwm
{

/**
 * \~english
 * @brief A combined class of distance and weight. 
 * Instances of this class are usually constructed by providing pointers to CGwmDistance and CGwmWeight.
 * In the construct function, instances of types CGwmDistance and CGwmWeight will be cloned.
 * This class provide the method CGwmSpatialWeight::weightVector() to calculate spatial weight directly.
 * 
 * If the distance and weight are set by pointers, this class will take the control of them, 
 * and when destructing the pointers will be deleted. 
 * If the distance and weight are set by references, this class will clone them. 
 * 
 * \~chinese
 * @brief 距离和权重的组合类。
 * 该类的实例通常用于提供指向 CGwmDistance 和 CGwmWeight 类型的指针。
 * 在构造函数中，类型 CGwmDistance 和 CGwmWeight 的实例将会被克隆。
 * 该类型也提供方法 CGwmSpatialWeight::weightVector() 用于直接计算空间权重。
 * 
 * 如果距离和权重通过指针设置，那么该类对象将会取得他们的控制权，并在析构的时候释放资源。
 * 如果距离和权重通过引用设置，那么该类对象将会克隆他们。
 */
class SpatialWeight
#ifdef ENABLE_CUDA
    : public ISpatialCudaEnabled
#endif
{
public:

    /**
     * \~english
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * \~chinese
     * @brief 构造一个新的 CGwmSpatialWeight 对象。
     */
    SpatialWeight() {}

    /**
     * \~english
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * @param weight Pointer to a weight configuration.
     * @param distance Pointer to distance configuration.
     * 
     * \~chinese
     * @brief 构造一个新的 CGwmSpatialWeight 对象。
     * 
     * @param weight 指向权重配置的指针。
     * @param distance 指向距离配置的指针。
     */
    SpatialWeight(const Weight* weight, const Distance* distance)
    {
        mWeight = weight->clone();
        mDistance = distance->clone();
    }

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * @param weight Reference to a weight configuration.
     * @param distance Reference to distance configuration.
     * 
     * \~chinese
     * @brief 构造一个新的 CGwmSpatialWeight 对象。
     * 
     * @param weight 指向权重配置的引用。
     * @param distance 指向距离配置的引用。
     */
    SpatialWeight(const Weight& weight, const Distance& distance)
    {
        mWeight = weight.clone();
        mDistance = distance.clone();
    }

    /**
     * \~english
     * @brief Copy construct a new CGwmSpatialWeight object.
     * 
     * @param spatialWeight Reference to the object to copy from.
     * 
     * \~chinese
     * @brief 复制构造一个新的 CGwmSpatialWeight 对象。
     * 
     * @param spatialWeight 被复制对象的引用。
     */
    SpatialWeight(const SpatialWeight& spatialWeight)
    {
        mWeight = spatialWeight.mWeight->clone();
        mDistance = spatialWeight.mDistance->clone();
    }

    /**
     * @brief Move construct a new CGwmSpatialWeight object.
     * 
     * @param other Reference to the object to move from.
     * 
     * \~chinese
     * @brief 移动构造一个新的 CGwmSpatialWeight 对象。
     * 
     * @param other 被移动对象的引用。
     */
    SpatialWeight(SpatialWeight&& other)
    {
        mWeight = other.mWeight;
        mDistance = other.mDistance;
        
        other.mWeight = nullptr;
        other.mDistance = nullptr;
    }

    /**
     * \~english
     * @brief Destroy the CGwmSpatialWeight object.
     * 
     * \~chinese
     * @brief 销毁 CGwmSpatialWeight 对象。
     */
    virtual ~SpatialWeight();

    /**
     * \~english
     * @brief Get the pointer to CGwmSpatialWeight::mWeight .
     * 
     * @return Pointer to CGwmSpatialWeight::mWeight .
     * 
     * \~chinese
     * @brief 获得 CGwmSpatialWeight::mWeight 的指针。
     * 
     * @return 指针 CGwmSpatialWeight::mWeight 。
     */
    Weight *weight() const
    {
        return mWeight;
    }

    /**
     * \~english
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Pointer to CGwmWeight instance. 
     * Control of this pointer will be taken, and it will be deleted when destructing.
     * 
     * \~chinese
     * @brief 设置 CGwmSpatialWeight::mWeight 指针所指向的对象。
     * 
     * @param weight 指向 CGwmWeight 实例的指针。
     * 获取该指针的控制权，并在类对象析构时释放该指针所指向的对象。
     */
    void setWeight(Weight *weight)
    {
        if (weight && weight != mWeight)
        {
            if (mWeight) delete mWeight;
            mWeight = weight->clone();
        }
    }

    /**
     * \~english
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Reference to CGwmWeight instance.
     * This object will be cloned.
     * 
     * \~chinese
     * @brief 设置 CGwmSpatialWeight::mWeight 指针所指向的对象。
     * 
     * @param weight 指向 CGwmWeight 实例的指针。
     * 指针所指向的对象会被克隆。
     */
    void setWeight(Weight& weight)
    {
        if (mWeight) delete mWeight;
        mWeight = weight.clone();
    }

    /**
     * \~english
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Reference to CGwmWeight instance.
     * This object will be cloned.
     * 
     * \~chinese
     * @brief 设置 CGwmSpatialWeight::mWeight 指针所指向的对象。
     * 
     * @param weight 指向 CGwmWeight 实例的指针。
     * 指针所指向的对象会被克隆。
     */
    void setWeight(Weight&& weight)
    {
        if (mWeight) delete mWeight;
        mWeight = weight.clone();
    }

    /**
     * \~english
     * @brief Get the pointer to CGwmSpatialWeight::mWeight and cast it to required type.
     * 
     * @tparam T Type of return value. Only CGwmBandwidthWeight is allowed.
     * @return Casted pointer to CGwmSpatialWeight::mWeight.
     * 
     * \~chinese
     * @brief 获得指针 CGwmSpatialWeight::mWeight 并将其转换到所要求的类型 \p T 。
     * 
     * @tparam T 返回值的类型。只允许设置为 CGwmWeight 的派生类。
     * @return 转换后的 CGwmSpatialWeight::mWeight 指针。
     */
    template<typename T>
    T* weight() const { return nullptr; }

    /**
     * \~english
     * @brief Get the pointer to CGwmSpatialWeight::mDistance.
     * 
     * @return Pointer to CGwmSpatialWeight::mDistance.
     * 
     * \~chinese
     * @brief 获得指针 CGwmSpatialWeight::mDistance。
     * 
     * @return CGwmSpatialWeight::mDistance 指针。
     */
    Distance *distance() const
    {
        return mDistance;
    }

    /**
     * \~english
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Pointer to CGwmDistance instance. 
     * Control of this pointer will be taken, and it will be deleted when destructing.
     * 
     * \~chinese
     * @brief 设置 CGwmSpatialWeight::mDistance 指针所指向的对象。
     * 
     * @param distance Pointer to CGwmDistance instance. 
     * Con获取该指针的控制权，并在类对象析构时释放该指针所指向的对象。
     */
    void setDistance(Distance *distance)
    {
        if (distance && distance != mDistance)
        {
            if (mDistance) delete mDistance;
            mDistance = distance->clone();
        }
    }

    /**
     * \~english
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Reference to CGwmDistance instance.
     * This object will be cloned.
     * 
     * \~chinese
     * @brief 设置 CGwmSpatialWeight::mDistance 指针所指向的对象。
     * 
     * @param distance 指向 CGwmDistance 实例的指针。
     * 指针所指向的对象会被克隆。
     */
    void setDistance(Distance& distance)
    {
        if (mDistance) delete mDistance;
        mDistance = distance.clone();
    }

    /**
     * \~english
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Reference to CGwmDistance instance.
     * This object will be cloned.
     * 
     * \~chinese
     * @brief 设置 CGwmSpatialWeight::mDistance 指针所指向的对象。
     * 
     * @param distance 指向 CGwmDistance 实例的指针。
     * 指针所指向的对象会被克隆。
     */
    void setDistance(Distance&& distance)
    {
        if (mDistance) delete mDistance;
        mDistance = distance.clone();
    }

    /**
     * \~english
     * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to required type.
     * 
     * @tparam T Type of return value. Only CGwmCRSDistance and CGwmMinkwoskiDistance is allowed.
     * @return Casted pointer to CGwmSpatialWeight::mDistance.
     * 
     * \~chinese
     * @brief 获得指针 CGwmSpatialWeight::mDistance 并将其转换到所要求的类型 \p T 。
     * 
     * @tparam T 返回值的类型。只允许设置为 CGwmDistance 的派生类。
     * @return 转换后的 CGwmSpatialWeight::mDistance 指针。
     */
    template<typename T>
    T* distance() const { return nullptr; }

public:

    /**
     * \~english
     * @brief Override operator = for this class. 
     * This function will first delete the current CGwmSpatialWeight::mWeight and CGwmSpatialWeight::mDistance,
     * and then clone CGwmWeight and CGwmDistance instances according pointers of the right value. 
     * 
     * @param spatialWeight Reference to the right value.
     * @return Reference of this object.
     * 
     * \~chinese
     * @brief 重载的 \p = 运算符。
     * 该函数会先实方当前 CGwmSpatialWeight::mWeight 和 CGwmSpatialWeight::mDistance 所指向的对象，
     * 然后克隆右值传入的 CGwmWeight 和 CGwmDistance 实例。
     * 
     * @param spatialWeight 右值的引用。
     * @return 该对象的引用。
     */
    SpatialWeight& operator=(const SpatialWeight& spatialWeight);

    /**
     * \~english
     * @brief Override operator = for this class. 
     * This function will first delete the current CGwmSpatialWeight::mWeight and CGwmSpatialWeight::mDistance,
     * and then clone CGwmWeight and CGwmDistance instances according pointers of the right value. 
     * 
     * @param spatialWeight Right value reference to the right value.
     * @return Reference of this object.
     * 
     * \~chinese
     * @brief 重载的 \p = 运算符。
     * 该函数会先实方当前 CGwmSpatialWeight::mWeight 和 CGwmSpatialWeight::mDistance 所指向的对象，
     * 然后克隆右值传入的 CGwmWeight 和 CGwmDistance 实例。
     * 
     * @param spatialWeight 右值的引用。
     * @return 该对象的引用。
     */
    SpatialWeight& operator=(SpatialWeight&& spatialWeight);

public:

    /**
     * \~english
     * @brief Calculate the spatial weight vector from focused sample to other samples (including the focused sample itself).
     * 
     * @param focus Index of current sample.
     * @return vec The spatial weight vector from focused sample to other samples.
     * 
     * \~chinese
     * @brief 计算当前样本到其他样本的空间权重向量（包括当前样本自身）。
     * 
     * @param focus 当前样本的索引值。
     * @return vec 当前样本到其他所有样本的空间权重向量。
     */
    virtual arma::vec weightVector(arma::uword focus) const
    {
        return mWeight->weight(mDistance->distance(focus));
    }

#ifdef ENABLE_CUDA
    virtual cudaError_t prepareCuda(size_t gpuId) override
    {
        cudaError_t error;
        error = mDistance->prepareCuda(gpuId);
        if (error != cudaSuccess) return error;
        error = mWeight->prepareCuda(gpuId);
        return error;
    }

    virtual bool useCuda()
    {
        return mWeight->useCuda() || mDistance->useCuda();
    }

    virtual void setUseCuda(bool isUseCuda)
    {
        mWeight->setUseCuda(isUseCuda);
        mDistance->setUseCuda(isUseCuda);
    }

    /**
     * @brief \~english Calculate the spatial weight vector from focused sample to other samples (including the focused sample itself).
     * \~chinese 计算当前样本到其他样本的空间权重向量（包括当前样本自身）。
     * 
     * @param focus \~english Focused point's index. Require focus < total \~chinese 目标点索引，要求 focus 小于参数中的 total
     * @param d_dists \~english Output device pointer to distances \~chinese 指向输出距离的设备指针
     * @param d_weights \~english Device pointer to distances \~chinese 指向输出权重的设备指针
     * @return cudaError_t \~english CUDA error or success \~chinese CUDA 错误或成功 
     */
    virtual cudaError_t weightVector(arma::uword focus, double* d_dists, double* d_weights)
    {
        cudaError_t error;
        size_t elems = 0;
        error = mDistance->distance(focus, d_dists, &elems);
        if (error != cudaSuccess) return error;
        error = mWeight->weight(d_dists, d_weights, elems);
        return error;
    }
#endif

    /**
     * \~english
     * @brief Get whether this object is valid in geting weight vector.
     * 
     * @return true if this object is valid.
     * @return false if this object is invalid.
     * 
     * \~chinese
     * @brief 获取当前对象的设置是否合法。
     * 
     * @return true 如果当前对象合法。
     * @return false 如果当前对象不合法。
     */
    virtual bool isValid();

private:
    Weight* mWeight = nullptr;      //!< Pointer to weight configuration.
    Distance* mDistance = nullptr;  //!< Pointer to distance configuration.
};

/**
 * \~english
 * @brief Get the pointer to CGwmSpatialWeight::mWeight and cast it to CGwmBandwidthWeight type.
 * 
 * @return Casted pointer to CGwmSpatialWeight::mWeight.
 * 
 * \~chinese
 * @brief 获得指针 CGwmSpatialWeight::mWeight 并将其转换到所要求的类型 CGwmBandwidthWeight 。
 * 
 * @return 转换后的 CGwmSpatialWeight::mWeight 指针。
 */
template<>
inline BandwidthWeight* SpatialWeight::weight<BandwidthWeight>() const
{
    return static_cast<BandwidthWeight*>(mWeight);
}

/**
 * \~english
 * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to CGwmCRSDistance type.
 * 
 * @return Casted pointer to CGwmSpatialWeight::mDistance.
 * 
 * \~chinese
 * @brief 获得指针 CGwmSpatialWeight::mDistance 并将其转换到所要求的类型 CGwmCRSDistance 。
 * 
 * @return 转换后的 CGwmSpatialWeight::mDistance 指针。
 */
template<>
inline CRSDistance* SpatialWeight::distance<CRSDistance>() const
{
    return static_cast<CRSDistance*>(mDistance);
}

/**
 * \~english
 * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to CGwmCRSSTDistance type.
 * 
 * @return Casted pointer to CGwmSpatialWeight::mDistance.
 * 
 * \~chinese
 * @brief 获得指针 CGwmSpatialWeight::mDistance 并将其转换到所要求的类型 CGwmCRSSTDistance 。
 * 
 * @return 转换后的 CGwmSpatialWeight::mDistance 指针。
 */
template<>
inline CRSSTDistance* SpatialWeight::distance<CRSSTDistance>() const
{
    return static_cast<CRSSTDistance*>(mDistance);
}

/**
 * \~english
 * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to CGwmMinkwoskiDistance type.
 * 
 * @return Casted pointer to CGwmSpatialWeight::mDistance.
 * 
 * \~chinese
 * @brief 获得指针 CGwmSpatialWeight::mDistance 并将其转换到所要求的类型 CGwmMinkwoskiDistance 。
 * 
 * @return 转换后的 CGwmSpatialWeight::mDistance 指针。
 */
template<>
inline MinkwoskiDistance* SpatialWeight::distance<MinkwoskiDistance>() const
{
    return static_cast<MinkwoskiDistance*>(mDistance);
}

/**
 * \~english
 * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to CGwmDMatDistance type.
 * 
 * @return Casted pointer to CGwmSpatialWeight::mDistance.
 * 
 * \~chinese
 * @brief 获得指针 CGwmSpatialWeight::mDistance 并将其转换到所要求的类型 CGwmDMatDistance 。
 * 
 * @return 转换后的 CGwmSpatialWeight::mDistance 指针。
 */
template<>
inline DMatDistance* SpatialWeight::distance<DMatDistance>() const
{
    return static_cast<DMatDistance*>(mDistance);
}

/**
 * \~english
 * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to CGwmOneDimDistance type.
 * 
 * @return Casted pointer to CGwmSpatialWeight::mDistance.
 * 
 * \~chinese
 * @brief 获得指针 CGwmSpatialWeight::mDistance 并将其转换到所要求的类型 CGwmOneDimDistance 。
 * 
 * @return 转换后的 CGwmSpatialWeight::mDistance 指针。
 */
template<>
inline OneDimDistance* SpatialWeight::distance<OneDimDistance>() const
{
    return static_cast<OneDimDistance*>(mDistance);
}

}

#endif // SPATIALWEIGHT_H
