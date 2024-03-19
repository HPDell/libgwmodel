#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRLocalCollinearity.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("LocalCollinearityGWR: basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GWRLocalCollinearity algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    //algorithm.setLambdaAdjust(true);
    REQUIRE_NOTHROW(algorithm.fit());

    RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2461.5654565, 1e-8));
    REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2464.60025589, 1e-8));
    REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632043, 1e-8));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341722, 1e-8));

    REQUIRE(algorithm.hasIntercept() == true);
}

TEST_CASE("LocalCollinearityGWR: adaptive bandwidth autoselection of with CV")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GWRLocalCollinearity algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);

    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(GWRLocalCollinearity::BandwidthSelectionCriterionType::CV);
    
    REQUIRE_NOTHROW(algorithm.fit());

    size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 67);
}

TEST_CASE("LocalCollinearityGWR: ")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GWRLocalCollinearity algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setLambdaAdjust(true);
    REQUIRE_NOTHROW(algorithm.fit());

    RegressionDiagnostic diagnostic = algorithm.diagnostic();
    /*REQUIRE_THAT(diagnostic.AIC, Catch::MatchersWithinAbs(2461.565456, 1e-6));
    REQUIRE_THAT(diagnostic.AICc, Catch::MatchersWithinAbs(2464.600255, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::MatchersWithinAbs(0.708010632044736, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::MatchersWithinAbs(0.674975341723766, 1e-6));*/

    
}
/*
#ifdef ENABLE_OPENMP
TEST_CASE("LocalCollinearityGWR: multithread basic flow")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    GWRLocalCollinearity algorithm;
    algorithm.setCoords(londonhp100_coord);
    algorithm.setDependentVariable(y);
    algorithm.setIndependentVariables(x);
    algorithm.setSpatialWeight(spatial);
    algorithm.setHasHatMatrix(true);
    algorithm.setIsAutoselectBandwidth(true);
    algorithm.setBandwidthSelectionCriterion(GWRLocalCollinearity::BandwidthSelectionCriterionType::CV);
    algorithm.setParallelType(ParallelType::OpenMP);
    algorithm.setOmpThreadNum(6);
    REQUIRE_NOTHROW(algorithm.fit());


    double bw = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
    REQUIRE(bw == 67.0);

    RegressionDiagnostic diagnostic = algorithm.diagnostic();
    REQUIRE_THAT(diagnostic.AIC, Catch::MatchersWithinAbs(2458.2472656218, 1e-6));
    REQUIRE_THAT(diagnostic.AICc, Catch::MatchersWithinAbs(2459.743757379, 1e-6));
    REQUIRE_THAT(diagnostic.RSquare, Catch::MatchersWithinAbs(0.68733847, 1e-6));
    REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::MatchersWithinAbs(0.66436287, 1e-6));
}
#endif*/

TEST_CASE("LcGWR: cancel")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }

    CRSDistance distance(false);
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        , ParallelType::OpenMP
#endif // ENABLE_OPENMP     
    };
    auto parallel = GENERATE_REF(values(parallel_list));

    SECTION("fit")
    {
        auto stage = GENERATE(as<std::string>{}, "bandwidthSize", "fit");
        auto progress = GENERATE(0, 10);
        INFO("Settings: " << "Parallel:" << parallel << ", Stage:" << stage << ", " << progress);

        auto telegram = make_unique<TerminateCheckTelegram>(stage, progress);
        GWRLocalCollinearity algorithm;
        algorithm.setTelegram(std::move(telegram));
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setHasHatMatrix(true);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRLocalCollinearity::BandwidthSelectionCriterionType::CV);
        algorithm.setParallelType(parallel);
        algorithm.setOmpThreadNum(6);
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.status() == Status::Terminated);
    }
    
}
