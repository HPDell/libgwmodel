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

#include <mpi.h>

using namespace std;
using namespace arma;
using namespace gwm;


TEST_CASE("MGWR: basic flow")
{
    int iProcess, nProcess;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &iProcess);

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
        ParallelType::MPI_Serial,
#ifdef ENABLE_OPENMP
        ParallelType::MPI_MP,
#endif // ENABLE_OPENMP
#ifdef ENABLE_CUDA
        ParallelType::MPI_CUDA
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
        // algorithm.setTelegram(std::move(make_unique<Logger>()));
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
        algorithm.setWorkerId(iProcess);
        algorithm.setWorkerNum(nProcess);
        REQUIRE_NOTHROW(algorithm.fit());

        if (iProcess == 0)
        {
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
        // algorithm.setTelegram(std::move(make_unique<Logger>()));
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
        algorithm.setWorkerId(iProcess);
        algorithm.setWorkerNum(nProcess);
        REQUIRE_NOTHROW(algorithm.fit());

        if (iProcess == 0)
        {
            const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
            REQUIRE_THAT(spatialWeights[0].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(45, 0.1));
            REQUIRE_THAT(spatialWeights[1].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));
            REQUIRE_THAT(spatialWeights[2].weight<BandwidthWeight>()->bandwidth(), Catch::Matchers::WithinAbs(98, 0.1));

            RegressionDiagnostic diagnostic = algorithm.diagnostic();
            REQUIRE_THAT(diagnostic.AICc, Catch::Matchers::WithinAbs(2437.935218705351, 1e-6));
            REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.7486787930045755, 1e-6));
            REQUIRE_THAT(diagnostic.RSquareAdjust, Catch::Matchers::WithinAbs(0.7118919517893492, 1e-6));
        }
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
        // algorithm.setTelegram(std::move(make_unique<Logger>()));
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
        algorithm.setWorkerId(iProcess);
        algorithm.setWorkerNum(nProcess);
        REQUIRE_NOTHROW(algorithm.fit());

        if (iProcess == 0)
        {
            const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
            REQUIRE(spatialWeights[0].weight<BandwidthWeight>()->bandwidth() == 35);
            REQUIRE(spatialWeights[1].weight<BandwidthWeight>()->bandwidth() == 98);
            REQUIRE(spatialWeights[2].weight<BandwidthWeight>()->bandwidth() == 98);
            RegressionDiagnostic diagnostic = algorithm.diagnostic();
            REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));
            REQUIRE(algorithm.hasIntercept() == true);
        }
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
        // algorithm.setTelegram(std::move(make_unique<Logger>()));
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
        algorithm.setWorkerId(iProcess);
        algorithm.setWorkerNum(nProcess);
        REQUIRE_NOTHROW(algorithm.fit());

        if (iProcess == 0)
        {
            const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
            REQUIRE(spatialWeights[0].weight<BandwidthWeight>()->bandwidth() == 35);
            REQUIRE(spatialWeights[1].weight<BandwidthWeight>()->bandwidth() == 98);
            REQUIRE(spatialWeights[2].weight<BandwidthWeight>()->bandwidth() == 98);

            RegressionDiagnostic diagnostic = algorithm.diagnostic();
            REQUIRE_THAT(diagnostic.RSquare, Catch::Matchers::WithinAbs(0.757377391669, 1e-6));
        }
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
        // algorithm.setTelegram(std::move(make_unique<Logger>()));
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

        if (iProcess == 0)
        {
            const vector<SpatialWeight>& spatialWeights = algorithm.spatialWeights();
            REQUIRE(spatialWeights[0].weight<BandwidthWeight>()->bandwidth() == 52);
            REQUIRE(spatialWeights[1].weight<BandwidthWeight>()->bandwidth() == 99);
            REQUIRE(spatialWeights[2].weight<BandwidthWeight>()->bandwidth() == 99);
        }
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run( argc, argv );
    MPI_Finalize();
    return result;
}
