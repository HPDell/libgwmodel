#include "CGwmSpatialAlgorithm.h"

CGwmSpatialAlgorithm::CGwmSpatialAlgorithm()
{

}

CGwmSpatialAlgorithm::~CGwmSpatialAlgorithm()
{

}

bool CGwmSpatialAlgorithm::isValid()
{
    if (mSourceLayer == nullptr)
        return false;
    
    return true;
}