#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/CGwmGWRMultiscale.h"

#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include "gwmodelpp/spatialweight/CGwmBandwidthWeight.h"
#include "gwmodelpp/spatialweight/CGwmSpatialWeight.h"
#include "londonhp100.h"

using namespace std;
using namespace arma;
using namespace gwm;


TEST_CASE("MGWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, false, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(4623.78, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(12665.70, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(12665.70, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2437.09277417389, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.744649364494, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.712344894394, 1e-6));

    REQUIRE(algorithm.hasIntercept() == true);
}

TEST_CASE("MGWR: basic flow without hat matrix")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(false);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth() == 35);
    REQUIRE(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth() == 98);
    REQUIRE(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth() == 98);

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));

    REQUIRE(algorithm.hasIntercept() == true);
}

TEST_CASE("MGWR: adaptive bandwidth autoselection of with AIC")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Initial);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::AIC);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmGWRMultiscale::BackFittingCriterionType::dCVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(45, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2437.935218705351, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7486787930045755, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.7118919517893492, 1e-6));
}


TEST_CASE("MGWR: adaptive bandwidth autoselection of with CV")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmGWRMultiscale::BackFittingCriterionType::dCVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(35, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2438.256543499568, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391648, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.715598248202, 1e-6));
}

#ifdef ENABLE_OPENMP
TEST_CASE("MGWR: basic flow with CVR")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(36, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Initial);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmGWRMultiscale::BackFittingCriterionType::CVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    algorithm.setParallelType(ParallelType::OpenMP);
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth() == 35);
    REQUIRE(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth() == 98);
    REQUIRE(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth() == 98);

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2438.256543496552, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.715598248225, 1e-6));
}
#endif

#ifdef ENABLE_OPENMP
TEST_CASE("MGWR: basic flow (multithread)")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<CGwmSpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<CGwmGWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    vector<CGwmGWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
    for (size_t i = 0; i < nVar; i++)
    {
        CGwmCRSDistance distance;
        CGwmBandwidthWeight bandwidth(0, true, CGwmBandwidthWeight::Bisquare);
        spatials.push_back(CGwmSpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(CGwmGWRMultiscale::BandwidthInitilizeType::Null);
        bandwidthSelectionApproach.push_back(CGwmGWRMultiscale::BandwidthSelectionCriterionType::CV);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    CGwmGWRMultiscale algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeights(spatials);
    algorithm.setHasHatMatrix(true);
    algorithm.setCriterionType(CGwmGWRMultiscale::BackFittingCriterionType::dCVR);
    algorithm.setPreditorCentered(preditorCentered);
    algorithm.setBandwidthInitilize(bandwidthInitialize);
    algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
    algorithm.setBandwidthSelectRetryTimes(5);
    algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
    algorithm.setParallelType(ParallelType::OpenMP);
    REQUIRE_NOTHROW(algorithm.fit());

    const vector<CGwmSpatialWeight>& spatialWeights = algorithm.spatialWeights();
    REQUIRE_THAT(spatialWeights[0].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(35, 0.1));
    REQUIRE_THAT(spatialWeights[1].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));
    REQUIRE_THAT(spatialWeights[2].weight<CGwmBandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));

    GwmRegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2438.256543499568, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391648, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.715598248202, 1e-6));
}
#endif
