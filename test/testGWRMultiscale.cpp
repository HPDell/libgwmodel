#define CATCH_CONFIG_MAIN

#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRMultiscale.h"

#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"
#include "FileTelegram.h"

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

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));
    uword nVar = 3;

    const initializer_list<ParallelType> parallelTypes = {
        // ParallelType::SerialOnly,
#ifdef ENABLE_OPENMP
        // ParallelType::OpenMP,
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        ParallelType::CUDA
#endif // ENABLE_CUDA
    };

    SECTION("optim bw cv | null init bw | fixed | bisquare kernel | with hatmatrix")
    {
        auto parallel = GENERATE_REF(values(parallelTypes));
        INFO("Parallel type: " << ParallelTypeDict.at(parallel));

        vector<SpatialWeight> spatials;
        vector<bool> preditorCentered;
        vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(0, false, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            preditorCentered.push_back(i != 0);
            bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWRMultiscale::BandwidthSelectionCriterionType::CV);
        }

        GWRMultiscale algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(true);
        algorithm.setPreditorCentered(preditorCentered);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
        algorithm.setParallelType(parallel);
        REQUIRE_NOTHROW(algorithm.fit());

        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE_THAT(spatialWeights[0].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(4623.78, 0.1));
        REQUIRE_THAT(spatialWeights[1].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(12665.70, 0.1));
        REQUIRE_THAT(spatialWeights[2].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(12665.70, 0.1));

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2437.09277417389, 1e-6));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.744649364494, 1e-6));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.712344894394, 1e-6));

        REQUIRE(algorithm.hasIntercept() == true);
    }

    SECTION("optim bw AIC | null init bw | adaptive | bisquare kernel | with hatmatrix")
    {
        auto parallel = GENERATE_REF(values(parallelTypes));
        INFO("Parallel type: " << ParallelTypeDict.at(parallel));

        vector<SpatialWeight> spatials;
        vector<bool> preditorCentered;
        vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            preditorCentered.push_back(i != 0);
            bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Initial);
            bandwidthSelectionApproach.push_back(GWRMultiscale::BandwidthSelectionCriterionType::AIC);
        }

        GWRMultiscale algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(true);
        algorithm.setCriterionType(GWRMultiscale::BackFittingCriterionType::dCVR);
        algorithm.setPreditorCentered(preditorCentered);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setBandwidthSelectRetryTimes(5);
        algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
        algorithm.setParallelType(parallel);
        REQUIRE_NOTHROW(algorithm.fit());

        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE_THAT(spatialWeights[0].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(45, 0.1));
        REQUIRE_THAT(spatialWeights[1].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));
        REQUIRE_THAT(spatialWeights[2].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2437.935218705351, 1e-6));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7486787930045755, 1e-6));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.7118919517893492, 1e-6));
    }

    SECTION("optim bw cv | null init bw | adaptive | bisquare kernel | without hatmatrix")
    {
        auto parallel = GENERATE_REF(values(parallelTypes));
        INFO("Parallel type: " << ParallelTypeDict.at(parallel));
        
        vector<SpatialWeight> spatials;
        vector<bool> preditorCentered;
        vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(0, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            preditorCentered.push_back(i != 0);
            bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWRMultiscale::BandwidthSelectionCriterionType::CV);
        }

        GWRMultiscale algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(false);
        algorithm.setPreditorCentered(preditorCentered);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
        algorithm.setParallelType(parallel);
        REQUIRE_NOTHROW(algorithm.fit());
        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE(spatialWeights[0].weight<BandwidthWeight>()->bandwidth() == 35);
        REQUIRE(spatialWeights[1].weight<BandwidthWeight>()->bandwidth() == 98);
        REQUIRE(spatialWeights[2].weight<BandwidthWeight>()->bandwidth() == 98);
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));
        REQUIRE(algorithm.hasIntercept() == true);
    }

    SECTION("optim bw cv | null init bw | adaptive | bisquare | CVR | without hatmatrix")
    {
        auto parallel = GENERATE_REF(values(parallelTypes));
        INFO("Parallel type: " << ParallelTypeDict.at(parallel));
        
        vector<SpatialWeight> spatials;
        vector<bool> preditorCentered;
        vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            preditorCentered.push_back(i != 0);
            bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Initial);
            bandwidthSelectionApproach.push_back(GWRMultiscale::BandwidthSelectionCriterionType::CV);
        }

        GWRMultiscale algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(true);
        algorithm.setCriterionType(GWRMultiscale::BackFittingCriterionType::CVR);
        algorithm.setPreditorCentered(preditorCentered);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setBandwidthSelectRetryTimes(5);
        algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
        algorithm.setParallelType(parallel);
        REQUIRE_NOTHROW(algorithm.fit());

        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE(spatialWeights[0].weight<BandwidthWeight>()->bandwidth() == 35);
        REQUIRE(spatialWeights[1].weight<BandwidthWeight>()->bandwidth() == 98);
        REQUIRE(spatialWeights[2].weight<BandwidthWeight>()->bandwidth() == 98);

        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2438.256543496552, 1e-6));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.715598248225, 1e-6));
    }

    SECTION("optim bw cv | null init bw | adaptive | bisquare kernel | with hatmatrix | with bounds")
    {
        vector<SpatialWeight> spatials;
        vector<bool> preditorCentered;
        vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            CRSDistance distance;
            BandwidthWeight bandwidth(36, true, BandwidthWeight::Bisquare);
            spatials.push_back(SpatialWeight(&bandwidth, &distance));
            preditorCentered.push_back(i != 0);
            bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Null);
            bandwidthSelectionApproach.push_back(GWRMultiscale::BandwidthSelectionCriterionType::CV);
        }

        GWRMultiscale algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(true);
        algorithm.setCriterionType(GWRMultiscale::BackFittingCriterionType::CVR);
        algorithm.setPreditorCentered(preditorCentered);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setBandwidthSelectRetryTimes(5);
        algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
        algorithm.setParallelType(ParallelType::SerialOnly);
        algorithm.setGoldenLowerBounds(50);
        algorithm.setGoldenUpperBounds(100);
        REQUIRE_NOTHROW(algorithm.fit());

        const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
        REQUIRE(spatialWeights[0].weight<BandwidthWeight>()->bandwidth() == 52);
        REQUIRE(spatialWeights[1].weight<BandwidthWeight>()->bandwidth() == 99);
        REQUIRE(spatialWeights[2].weight<BandwidthWeight>()->bandwidth() == 99);
    }
}

TEST_CASE("Multiscale GWR: cancel")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    uword nVar = 3;
    vector<SpatialWeight> spatials;
    vector<bool> preditorCentered;
    vector<GWRMultiscale::BandwidthInitilizeType> bandwidthInitialize;
    for (size_t i = 0; i < nVar; i++)
    {
        CRSDistance distance;
        BandwidthWeight bandwidth(0, false, BandwidthWeight::Bisquare);
        spatials.push_back(SpatialWeight(&bandwidth, &distance));
        preditorCentered.push_back(i != 0);
        bandwidthInitialize.push_back(GWRMultiscale::BandwidthInitilizeType::Null);
    }

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_data.n_rows), londonhp100_data.cols(uvec({1, 3})));

    vector<pair<string, size_t>> fit_stages = {
        make_pair("bandwidthSizeCriterionVar", 0),
        make_pair("bandwidthSizeCriterionVar", 10),
        make_pair("bandwidthSizeCriterionAll", 0),
        make_pair("bandwidthSizeCriterionAll", 10),
        make_pair("fitAll", 0),
        make_pair("fitAll", 10),
        make_pair("fitVar", 0),
        make_pair("fitVar", 10)
    };

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        , ParallelType::OpenMP
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        , ParallelType::CUDA
#endif // ENABLE_CUDA
    };
    auto parallel = GENERATE_REF(values(parallel_list));
    
    auto stage = GENERATE(as<std::string>{}, "bandwidthSizeCriterionVar", "bandwidthSizeCriterionAll", "fitAll", "fitVar");
    auto progress = GENERATE(0, 10);
    auto bandwidthCriterion = GENERATE(GWRMultiscale::BandwidthSelectionCriterionType::CV, GWRMultiscale::BandwidthSelectionCriterionType::AIC);

    SECTION("fit")
    {
        INFO("Parallel:" << ParallelTypeDict.at(parallel) << ", BandwidthCriterion:" << bandwidthCriterion << ", Stage:" << stage << "(" << progress << ")");
        vector<GWRMultiscale::BandwidthSelectionCriterionType> bandwidthSelectionApproach;
        for (size_t i = 0; i < nVar; i++)
        {
            bandwidthSelectionApproach.push_back(bandwidthCriterion);
        }

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWRMultiscale algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeights(spatials);
        algorithm.setHasHatMatrix(true);
        algorithm.setPreditorCentered(preditorCentered);
        algorithm.setBandwidthInitilize(bandwidthInitialize);
        algorithm.setBandwidthSelectionApproach(bandwidthSelectionApproach);
        algorithm.setBandwidthSelectThreshold(vector(3, 1e-5));
        algorithm.setParallelType(parallel);
        algorithm.setOmpThreadNum(6);
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.status() == Status::Terminated);
    }

}
