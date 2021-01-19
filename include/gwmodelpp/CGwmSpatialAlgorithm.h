#ifndef CGWMSPATIALALGORITHM_H
#define CGWMSPATIALALGORITHM_H

#include "CGwmAlgorithm.h"
#include "CGwmSimpleLayer.h"

/**
 * @brief Abstract spatial algorithm class. 
 * This class cannot been constructed.
 * It defines some interface commonly used in spatial algorithms:
 * 
 * - Getter and setter of source layer.
 * - Getter and setter of result layer.
 * - Check if configuration is valid.
 * 
 * Pointer of this type can be put to these classes:
 * 
 * - CGwmGWRBasic
 * - CGwmGWSS
 * 
 */
class CGwmSpatialAlgorithm : public CGwmAlgorithm
{
public:

    /**
     * @brief Construct a new CGwmSpatialAlgorithm object.
     */
    CGwmSpatialAlgorithm();

    /**
     * @brief Destroy the CGwmSpatialAlgorithm object.
     */
    virtual ~CGwmSpatialAlgorithm();

public:

    /**
     * @brief Get the CGwmSpatialAlgorithm::mSourceLayer object.
     * 
     * @return CGwmSpatialAlgorithm::mSourceLayer.
     */
    CGwmSimpleLayer* sourceLayer() const;

    /**
     * @brief Set the CGwmSpatialAlgorithm::mSourceLayer object.
     * 
     * Use gwmodel_set_gwr_source_layer() to set this property to CGwmGWRBasic in shared build.
     * 
     * Use gwmodel_set_gwss_source_layer() to set this property to CGwmGWSS in shared build.
     * 
     * @param layer Pointer to source layer.
     */
    void setSourceLayer(CGwmSimpleLayer* layer);
    
    /**
     * @brief Get the CGwmSpatialAlgorithm::mResultLayer object .
     * 
     * Use gwmodel_get_gwr_result_layer() to get this property from CGwmGWRBasic in shared build.
     * 
     * Use gwmodel_get_gwss_result_layer() to get this property from CGwmGWSS in shared build.
     * 
     * @return CGwmSpatialAlgorithm::mResultLayer.
     */
    CGwmSimpleLayer* resultLayer() const;

    /**
     * @brief Set the CGwmSpatialAlgorithm::mResultLayer object
     * 
     * @param layer Pointer to result layer.
     */
    void setResultLayer(CGwmSimpleLayer* layer);

public:

    /**
     * @brief Check whether the algorithm's configuration is valid. 
     * 
     * @return true if the algorithm's configuration is valid.
     * @return false if the algorithm's configuration is invalid.
     */
    virtual bool isValid() = 0;

protected:
    CGwmSimpleLayer* mSourceLayer = nullptr;    //!< Pointer to source layer.
    CGwmSimpleLayer* mResultLayer = nullptr;    //!< Pointer to result layer.
};

inline CGwmSimpleLayer* CGwmSpatialAlgorithm::sourceLayer() const
{
    return mSourceLayer;
}

inline void CGwmSpatialAlgorithm::setSourceLayer(CGwmSimpleLayer* layer)
{
    mSourceLayer = layer;
}

inline CGwmSimpleLayer* CGwmSpatialAlgorithm::resultLayer() const
{
    return mResultLayer;
}

inline void CGwmSpatialAlgorithm::setResultLayer(CGwmSimpleLayer* layer)
{
    mResultLayer = layer;
}


#endif  // CGWMSPATIALALGORITHM_H