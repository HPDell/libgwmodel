#include <catch2/catch_all.hpp>
#include <armadillo>
#include <iostream>

#include <mpi.h>
#include "gwmodelpp/utils/armampi.h"

using namespace std;
using namespace arma;

TEST_CASE("mat mul mpi")
{
    int ip, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &ip);

    uword n = 100;
    uword size = n / np + (n % np == 0 ? 0 : 1);
    uword from = ip * size, to = min(from + size, n);

    mat A_all, B_all, C_all;
    if (ip == 0)
    {
        A_all = mat(n, n, arma::fill::randn);
        B_all = mat(n, n, arma::fill::randn);
        C_all = A_all * B_all;
    }

    mat A, B, C;
    A = A_all.rows(from, to - 1);
    B = B_all.rows(from, to - 1);
    printf("begin mat mul\n");
    mat_mul_mpi(A, B, C, ip, np, size);
    printf("end mat mul\n");

    if (ip == 0)
    {
        mat C_gather(n, n);
        C_gather.rows(from, to - 1) = C;
        uvec received(np, fill::zeros);
        received(0) = 1;
        auto bufsize = size * n;
        double *buf = new double[size * n];
        while (!all(received == 1))
        {
            MPI_Status status;
            MPI_Recv(buf, bufsize, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            received(status.MPI_SOURCE) = 1;
            uword ifrom = status.MPI_SOURCE * size, ito = min(ifrom + size, n), irange = ito - ifrom; 
            C_gather.rows(ifrom, ito - 1) = mat(buf, irange, n);
        }
        REQUIRE(approx_equal(C_gather, C_all, "absdiff", 1e-6));
    }
    else
    {
        MPI_Send(C.memptr(), C.n_elem, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run( argc, argv );
    MPI_Finalize();
    return result;
}