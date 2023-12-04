#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include <vector>
#include <string>
#include <armadillo>
#include "gwmodelpp/GWRBasic.h"
#include "gwmodelpp/spatialweight/CRSDistance.h"
#include "gwmodelpp/spatialweight/BandwidthWeight.h"
#include "gwmodelpp/spatialweight/SpatialWeight.h"
#include "londonhp100.h"
#include "TerminateCheckTelegram.h"
#include "FileTelegram.h"

using namespace std;
using namespace arma;
using namespace gwm;

TEST_CASE("BasicGWR: LondonHP")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }
    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));

    const initializer_list<ParallelType> parallel_list = {
        ParallelType::SerialOnly
#ifdef ENABLE_OPENMP
        , ParallelType::OpenMP
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        , ParallelType::CUDA
#endif // ENABLE_CUDA
    };

    SECTION("adaptive bandwidth | no bandwidth optimization | no variable optimization") {
        auto parallel = GENERATE_REF(values(parallel_list));
        INFO("Parallel:" << ParallelTypeDict.at(parallel));

        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRBasic algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_CUDA
        if (parallel == ParallelType::CUDA)
        {
            algorithm.setGPUId(0);
            algorithm.setGroupSize(64);
        }
#endif // ENABLE_CUDA
        REQUIRE_NOTHROW(algorithm.fit());
        REQUIRE(algorithm.hasIntercept() == true);
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2436.60445730413, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2448.27206524754, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.708010632044736, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.674975341723766, 1e-8));
    }
    
    SECTION("fixed bandwidth | no bandwidth optimization | no variable optimization") {
        auto parallel = GENERATE_REF(values(parallel_list));
        INFO("Parallel:" << ParallelTypeDict.at(parallel));
        
        CRSDistance distance(false);
        BandwidthWeight bandwidth(5000, false, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRBasic algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        if (parallel == ParallelType::OpenMP)
        {
            algorithm.setOmpThreadNum(6);
        }
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        if (parallel == ParallelType::CUDA)
        {
            algorithm.setGPUId(0);
            algorithm.setGroupSize(64);
        }
#endif // ENABLE_CUDA
        REQUIRE_NOTHROW(algorithm.fit());
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2437.649574267587, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2447.676281164379, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.701466295457, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.673691594464, 1e-8));
    }

    SECTION("adaptive bandwidth | CV bandwidth optimization | no variable optimization") {
        auto parallel = GENERATE_REF(values(parallel_list));
        INFO("Parallel:" << ParallelTypeDict.at(parallel));
        
        CRSDistance distance(false);
        BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRBasic algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        if (parallel == ParallelType::OpenMP)
        {
            algorithm.setOmpThreadNum(6);
        }
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        if (parallel == ParallelType::CUDA)
        {
            algorithm.setGPUId(0);
            algorithm.setGroupSize(64);
        }
#endif // ENABLE_CUDA
        REQUIRE_NOTHROW(algorithm.fit());
        size_t bw = (size_t)algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 67);
    }
    
    SECTION("adaptive bandwidth | no bandwidth optimization | AIC variable optimization") {
        auto parallel = GENERATE_REF(values(parallel_list));
        INFO("Parallel:" << ParallelTypeDict.at(parallel));
        
        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);

        GWRBasic algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectIndepVars(true);
        algorithm.setIndepVarSelectionThreshold(3.0);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        if (parallel == ParallelType::OpenMP)
        {
            algorithm.setOmpThreadNum(6);
        }
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        if (parallel == ParallelType::CUDA)
        {
            algorithm.setGPUId(0);
            algorithm.setGroupSize(64);
        }
#endif // ENABLE_CUDA
        REQUIRE_NOTHROW(algorithm.fit());
        VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
        REQUIRE_THAT(criterions[0].first, Catch::Matchers::Equals(vector<size_t>({ 2 })));
        REQUIRE_THAT(criterions[0].second, Catch::Matchers::WithinAbs(2551.61359020599, 1e-8));
        REQUIRE_THAT(criterions[1].first, Catch::Matchers::Equals(vector<size_t>({ 3 })));
        REQUIRE_THAT(criterions[1].second, Catch::Matchers::WithinAbs(2551.30032201349, 1e-8));
        REQUIRE_THAT(criterions[2].first, Catch::Matchers::Equals(vector<size_t>({ 1 })));
        REQUIRE_THAT(criterions[2].second, Catch::Matchers::WithinAbs(2468.93236280013, 1e-8));
        REQUIRE_THAT(criterions[3].first, Catch::Matchers::Equals(vector<size_t>({ 1, 3 })));
        REQUIRE_THAT(criterions[3].second, Catch::Matchers::WithinAbs(2452.86447942033, 1e-8));
        REQUIRE_THAT(criterions[4].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2 })));
        REQUIRE_THAT(criterions[4].second, Catch::Matchers::WithinAbs(2450.59642666509, 1e-8));
        REQUIRE_THAT(criterions[5].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2, 3 })));
        REQUIRE_THAT(criterions[5].second, Catch::Matchers::WithinAbs(2452.80388934625, 1e-8));
        vector<size_t> selectedVariables = algorithm.selectedVariables();
        REQUIRE_THAT(selectedVariables, Catch::Matchers::Equals(vector<size_t>({1, 3})));
    }

    SECTION("adaptive bandwidth | CV bandwidth optimization | AIC variable optimization") {
        auto parallel = GENERATE_REF(values(parallel_list));
        INFO("Parallel:" << ParallelTypeDict.at(parallel));
        
        CRSDistance distance(false);
        BandwidthWeight bandwidth(36, true, BandwidthWeight::Gaussian);
        SpatialWeight spatial(&bandwidth, &distance);
        
        GWRBasic algorithm;
        algorithm.setCoords(londonhp100_coord);
        algorithm.setDependentVariable(y);
        algorithm.setIndependentVariables(x);
        algorithm.setSpatialWeight(spatial);
        algorithm.setIsAutoselectBandwidth(true);
        algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
        algorithm.setIsAutoselectIndepVars(true);
        algorithm.setIndepVarSelectionThreshold(3.0);
        algorithm.setParallelType(parallel);
#ifdef ENABLE_OPENMP
        if (parallel == ParallelType::OpenMP)
        {
            algorithm.setOmpThreadNum(6);
        }
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        if (parallel == ParallelType::CUDA)
        {
            algorithm.setGPUId(0);
            algorithm.setGroupSize(64);
        }
#endif // ENABLE_CUDA
        REQUIRE_NOTHROW(algorithm.fit());
        VariablesCriterionList criterions = algorithm.indepVarsSelectionCriterionList();
        REQUIRE_THAT(criterions[0].first, Catch::Matchers::Equals(vector<size_t>({ 2 })));
        REQUIRE_THAT(criterions[0].second, Catch::Matchers::WithinAbs(2551.61359020599, 1e-8));
        REQUIRE_THAT(criterions[1].first, Catch::Matchers::Equals(vector<size_t>({ 3 })));
        REQUIRE_THAT(criterions[1].second, Catch::Matchers::WithinAbs(2551.30032201349, 1e-8));
        REQUIRE_THAT(criterions[2].first, Catch::Matchers::Equals(vector<size_t>({ 1 })));
        REQUIRE_THAT(criterions[2].second, Catch::Matchers::WithinAbs(2468.93236280013, 1e-8));
        REQUIRE_THAT(criterions[3].first, Catch::Matchers::Equals(vector<size_t>({ 1, 3 })));
        REQUIRE_THAT(criterions[3].second, Catch::Matchers::WithinAbs(2452.86447942033, 1e-8));
        REQUIRE_THAT(criterions[4].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2 })));
        REQUIRE_THAT(criterions[4].second, Catch::Matchers::WithinAbs(2450.59642666509, 1e-8));
        REQUIRE_THAT(criterions[5].first, Catch::Matchers::Equals(vector<size_t>({ 1, 2, 3 })));
        REQUIRE_THAT(criterions[5].second, Catch::Matchers::WithinAbs(2452.80388934625, 1e-8));
        vector<size_t> selectedVariables = algorithm.selectedVariables();
        REQUIRE_THAT(selectedVariables, Catch::Matchers::Equals(vector<size_t>({1, 3})));
        double bw = algorithm.spatialWeight().weight<BandwidthWeight>()->bandwidth();
        REQUIRE(bw == 31.0);
        RegressionDiagnostic diagnostic = algorithm.diagnostic();
        REQUIRE_THAT(diagnostic.AIC, Catch::Matchers::WithinAbs(2435.8161441795, 1e-8));
        REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2445.49629974057, 1e-8));
        REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.706143867720706, 1e-8));
        REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.678982114793865, 1e-8));
    }

}


TEST_CASE("Basic GWR: cancel")
{
    mat londonhp100_coord, londonhp100_data;
    vector<string> londonhp100_fields;
    if (!read_londonhp100(londonhp100_coord, londonhp100_data, londonhp100_fields))
    {
        FAIL("Cannot load londonhp100 data.");
    }
    vec y = londonhp100_data.col(0);
    mat x = join_rows(ones(londonhp100_coord.n_rows), londonhp100_data.cols(1, 3));
    CRSDistance distance(false);
    BandwidthWeight bandwidth(0, true, BandwidthWeight::Gaussian);
    SpatialWeight spatial(&bandwidth, &distance);

    vector<pair<string, size_t>> fit_stages = {
        make_pair("indepVarsSelection", 0),
        make_pair("indepVarsSelection", 10),
        make_pair("bandwidthSize", 0),
        make_pair("bandwidthSize", 10),
        make_pair("fit", 0),
        make_pair("fit", 10)
    };

    vector<pair<string, size_t>> predict_stages = {
        make_pair("predict", 0),
        make_pair("predict", 10)
    };

    const initializer_list<ParallelType> parallel_types = {
#ifdef ENABLE_OPENMP
        ParallelType::OpenMP,
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        ParallelType::CUDA,
#endif // ENABLE_CUDA
        ParallelType::SerialOnly
    };
    auto parallel = GENERATE_REF(values(parallel_types));

    SECTION("fit | CV Bandwidth")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::CV);
            algorithm.setParallelType(parallel);
            switch (parallel)
            {
            case ParallelType::OpenMP:
                algorithm.setOmpThreadNum(8);
                break;
            case ParallelType::CUDA:
                algorithm.setGPUId(0);
                algorithm.setGroupSize(64);
            default:
                break;
            }
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

    SECTION("fit | AIC Bandwidth")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
            algorithm.setParallelType(parallel);
            switch (parallel)
            {
            case ParallelType::OpenMP:
                algorithm.setOmpThreadNum(8);
                break;
            case ParallelType::CUDA:
                algorithm.setGPUId(0);
                algorithm.setGroupSize(64);
            default:
                break;
            }
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

    SECTION("predict")
    {
        for (auto &&stage : fit_stages)
        {
            auto telegram = make_unique<TerminateCheckTelegram>(stage.first, stage.second);
            GWRBasic algorithm;
            algorithm.setTelegram(std::move(telegram));
            algorithm.setCoords(londonhp100_coord);
            algorithm.setDependentVariable(y);
            algorithm.setIndependentVariables(x);
            algorithm.setSpatialWeight(spatial);
            algorithm.setIsAutoselectBandwidth(true);
            algorithm.setIsAutoselectIndepVars(true);
            algorithm.setBandwidthSelectionCriterion(GWRBasic::BandwidthSelectionCriterionType::AIC);
            algorithm.setParallelType(parallel);
            switch (parallel)
            {
            case ParallelType::OpenMP:
                algorithm.setOmpThreadNum(8);
                break;
            case ParallelType::CUDA:
                algorithm.setGPUId(0);
                algorithm.setGroupSize(64);
            default:
                break;
            }
            REQUIRE_NOTHROW(algorithm.fit());
            REQUIRE_NOTHROW(algorithm.predict(londonhp100_coord));
            REQUIRE(algorithm.status() == Status::Terminated);
        }
    }

}

#ifdef ENABLE_CUDA
TEST_CASE("BasicGWR: Benchmark")
{
    size_t n = 50000, k = 3;
    mat x(n, k, fill::randn);
    vec u(n, fill::randu), v(n, fill::randu);
    vec beta0 = u + v;
    vec beta1 = u % u + v % v;
    vec beta2 = sin(u) + cos(v);
    mat beta = join_rows(beta0, beta1, beta2);
    vec epsilon(n, fill::randn);
    vec y = sum(x % beta, 1) + epsilon;
    mat coords = join_rows(u, v);
    CRSDistance distance(false);
    BandwidthWeight bw(0.2, false, BandwidthWeight::Gaussian);
    
    BENCHMARK("simulation | OpenMP")
    {
        SpatialWeight sw(&bw, &distance);

        GWRBasic algorithm(x, y, coords, sw);
        algorithm.setParallelType(ParallelType::OpenMP);
        algorithm.setOmpThreadNum(12);
        algorithm.fit();

        return algorithm.betas();
    };
    
    BENCHMARK("simulation | CUDA")
    {
        SpatialWeight sw(&bw, &distance);

        GWRBasic algorithm(x, y, coords, sw);
        algorithm.setParallelType(ParallelType::CUDA);
        algorithm.setGPUId(0);
        algorithm.setGroupSize(256);
        algorithm.fit();

        return algorithm.betas();
    };
}
#endif // ENABLE_CUDA
