#include "CGwmSpatialAlgorithm.h"

CGwmSpatialAlgorithm::CGwmSpatialAlgorithm()
{

}

CGwmSpatialAlgorithm::~CGwmSpatialAlgorithm()
{
    printf("~CGwmSpatialAlgorithm %lld");
}

bool CGwmSpatialAlgorithm::isValid()
{
    if (mSourceLayer == nullptr)
        return false;
    
    return true;
}