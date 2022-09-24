#include "CGwmSpatialMonoscaleAlgorithm.h"

CGwmSpatialMonoscaleAlgorithm::CGwmSpatialMonoscaleAlgorithm()
{

}

CGwmSpatialMonoscaleAlgorithm::~CGwmSpatialMonoscaleAlgorithm()
{

}

void CGwmSpatialMonoscaleAlgorithm::createDistanceParameter()
{
    if (mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::CRSDistance || 
        mSpatialWeight.distance()->type() == CGwmDistance::DistanceType::MinkwoskiDistance)
    {
        mSpatialWeight.distance()->makeParameter({
            mSourceLayer->points(),
            mSourceLayer->points()
        });
    }
}
