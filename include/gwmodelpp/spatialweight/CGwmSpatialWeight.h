#ifndef CGWMSPATIALWEIGHT_H
#define CGWMSPATIALWEIGHT_H

#include "gwmodelpp/spatialweight/CGwmWeight.h"
#include "gwmodelpp/spatialweight/CGwmDistance.h"

#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"

#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmMinkwoskiDistance.h"
#include "gwmodelpp/spatialweight/CGwmDMatDistance.h"

/**
 * @brief A combined class of distance and weight. 
 * Instances of this class are usually constructed by providing pointers to CGwmDistance and CGwmWeight.
 * In the construct function, instances will be cloned.
 * This class provide the method CGwmSpatialWeight::weightVector() to calculate spatial weight directly.
 * 
 * If the distance and weight are set by pointers, this class will take the control of them, 
 * and when destructing the pointers will be deleted. 
 * If the distance and weight are set by references, this class will clone them. 
 */
class CGwmSpatialWeight
{
public:

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     */
    CGwmSpatialWeight();

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * @param weight Pointer to a weight configuration.
     * @param distance Pointer to distance configuration.
     */
    CGwmSpatialWeight(CGwmWeight* weight, CGwmDistance* distance);

    /**
     * @brief Construct a new CGwmSpatialWeight object.
     * 
     * @param spatialWeight Reference to the object to copy from.
     */
    CGwmSpatialWeight(const CGwmSpatialWeight& spatialWeight);

    /**
     * @brief Destroy the CGwmSpatialWeight object.
     */
    ~CGwmSpatialWeight();

    /**
     * @brief Get the pointer to CGwmSpatialWeight::mWeight .
     * 
     * @return Pointer to CGwmSpatialWeight::mWeight .
     */
    CGwmWeight *weight() const;

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Pointer to CGwmWeight instance. 
     * Control of this pointer will be taken, and it will be deleted when destructing.
     */
    void setWeight(CGwmWeight *weight);

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Reference to CGwmWeight instance.
     * This object will be cloned.
     */
    void setWeight(CGwmWeight& weight);

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mWeight object.
     * 
     * @param weight Reference to CGwmWeight instance.
     * This object will be cloned.
     */
    void setWeight(CGwmWeight&& weight);

    /**
     * @brief Get the pointer to CGwmSpatialWeight::mWeight and cast it to required type.
     * 
     * @tparam T Type of return value. Only CGwmBandwidthWeight is allowed.
     * @return Casted pointer to CGwmSpatialWeight::mWeight.
     */
    template<typename T>
    T* weight() const { return nullptr; }

    /**
     * @brief Get the pointer to CGwmSpatialWeight::mDistance.
     * 
     * @return Pointer to CGwmSpatialWeight::mDistance.
     */
    CGwmDistance *distance() const;

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Pointer to CGwmDistance instance. 
     * Control of this pointer will be taken, and it will be deleted when destructing.
     */
    void setDistance(CGwmDistance *distance);

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Reference to CGwmDistance instance.
     * This object will be cloned.
     */
    void setDistance(CGwmDistance& distance);

    /**
     * @brief Set the pointer to CGwmSpatialWeight::mDistance object.
     * 
     * @param distance Reference to CGwmDistance instance.
     * This object will be cloned.
     */
    void setDistance(CGwmDistance&& distance);

    /**
     * @brief Get the pointer to CGwmSpatialWeight::mDistance and cast it to required type.
     * 
     * @tparam T Type of return value. Only CGwmCRSDistance and CGwmMinkwoskiDistance is allowed.
     * @return Casted pointer to CGwmSpatialWeight::mDistance.
     */
    template<typename T>
    T* distance() const { return nullptr; }

public:

    /**
     * @brief Override operator = for this class. 
     * This function will first delete the current CGwmSpatialWeight::mWeight and CGwmSpatialWeight::mDistance,
     * and then clone CGwmWeight and CGwmDistance instances according pointers of the right value. 
     * 
     * @param spatialWeight Reference to the right value.
     * @return Reference of this object.
     */
    CGwmSpatialWeight& operator=(const CGwmSpatialWeight& spatialWeight);

    /**
     * @brief Override operator = for this class. 
     * This function will first delete the current CGwmSpatialWeight::mWeight and CGwmSpatialWeight::mDistance,
     * and then clone CGwmWeight and CGwmDistance instances according pointers of the right value. 
     * 
     * @param spatialWeight Right value reference to the right value.
     * @return Reference of this object.
     */
    CGwmSpatialWeight& operator=(const CGwmSpatialWeight&& spatialWeight);

public:

    /**
     * @brief 
     * 
     * @param parameter 
     * @param focus 
     * @return vec 
     */
    virtual vec weightVector(DistanceParameter* parameter, uword focus);

    /**
     * @brief Get whether this object is valid in geting weight vector.
     * 
     * @return true if this object is valid.
     * @return false if this object is invalid.
     */
    virtual bool isValid();

private:
    CGwmWeight* mWeight = nullptr;      //!< Pointer to weight configuration.
    CGwmDistance* mDistance = nullptr;  //!< Pointer to distance configuration.
};

inline CGwmWeight *CGwmSpatialWeight::weight() const
{
    return mWeight;
}

template<>
inline CGwmBandwidthWeight* CGwmSpatialWeight::weight<CGwmBandwidthWeight>() const
{
    return static_cast<CGwmBandwidthWeight*>(mWeight);
}

inline void CGwmSpatialWeight::setWeight(CGwmWeight *weight)
{
    if (mWeight) delete mWeight;
    mWeight = weight;
}

inline void CGwmSpatialWeight::setWeight(CGwmWeight& weight)
{
    if (mWeight) delete mWeight;
    mWeight = weight.clone();
}

inline void CGwmSpatialWeight::setWeight(CGwmWeight&& weight)
{
    if (mWeight) delete mWeight;
    mWeight = weight.clone();
}

inline CGwmDistance *CGwmSpatialWeight::distance() const
{
    return mDistance;
}

template<>
inline CGwmCRSDistance* CGwmSpatialWeight::distance<CGwmCRSDistance>() const
{
    return static_cast<CGwmCRSDistance*>(mDistance);
}

template<>
inline CGwmMinkwoskiDistance* CGwmSpatialWeight::distance<CGwmMinkwoskiDistance>() const
{
    return static_cast<CGwmMinkwoskiDistance*>(mDistance);
}

template<>
inline CGwmDMatDistance* CGwmSpatialWeight::distance<CGwmDMatDistance>() const
{
    return static_cast<CGwmDMatDistance*>(mDistance);
}

inline void CGwmSpatialWeight::setDistance(CGwmDistance *distance)
{
    if (mDistance) delete mDistance;
    mDistance = distance;
}

inline void CGwmSpatialWeight::setDistance(CGwmDistance& distance)
{
    if (mDistance) delete mDistance;
    mDistance = distance.clone();
}

inline void CGwmSpatialWeight::setDistance(CGwmDistance&& distance)
{
    if (mDistance) delete mDistance;
    mDistance = distance.clone();
}

inline vec CGwmSpatialWeight::weightVector(DistanceParameter* parameter, uword focus)
{
    return mWeight->weight(mDistance->distance(parameter, focus));
}

#endif // CGWMSPATIALWEIGHT_H
