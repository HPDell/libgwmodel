/**
 * @file gwmodel.h
 * @author YigongHu (hu_yigong@whu.edu.cn)
 * @brief This file provide the headers of this library. 
 * If the library is built as a static library, this header includes all the C++ headers. 
 * If the library is built as a shared library, this header provides interface functions for the calling of C++ classes.
 * @version 0.1.0
 * @date 2020-10-08
 * 
 * @copyright Copyright (c) 2020
 * 
 * 
 */

#ifndef GWMODEL_H
#define GWMODEL_H

#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/Distance.h"
#include "gwmodelpp/spatialweight/DMatDistance.h"
#include "gwmodelpp/spatialweight/MinkwoskiDistance.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "gwmodelpp/spatialweight/Weight.h"

#include "gwmodelpp/Algorithm.h"
#include "gwmodelpp/BandwidthSelector.h"
#include "gwmodelpp/VariableForwardSelector.h"
#include "gwmodelpp/SpatialAlgorithm.h"
#include "gwmodelpp/SpatialMonoscaleAlgorithm.h"
#include "gwmodelpp/SpatialMultiscaleAlgorithm.h"
#include "gwmodelpp/GWRBase.h"
#include "gwmodelpp/GWRBasic.h"
#include "gwmodelpp/GWRGeneralized.h"
#include "gwmodelpp/GWRLocalCollinearity.h"
#include "gwmodelpp/GWRMultiscale.h"
#include "gwmodelpp/GWRRobust.h"
#include "gwmodelpp/GWRScalable.h"
#include "gwmodelpp/SDR.h"
#include "gwmodelpp/GWSS.h"
#include "gwmodelpp/GWPCA.h"

#endif  // GWMODEL_H