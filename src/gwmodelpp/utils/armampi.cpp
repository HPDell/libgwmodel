#include "gwmodelpp/utils/armampi.h"
#include <mpi.h>
#include <memory>

using namespace std;
using namespace arma;

void mat_mul_mpi(mat& a, mat& b, mat& c, const int ip, const int np, const size_t range)
{
    auto m = a.n_rows, k = a.n_cols;
    arma::uvec b_rows(np, arma::fill::zeros);
    MPI_Allgather(&b.n_rows, 1, MPI_UNSIGNED_LONG_LONG, b_rows.memptr(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    arma::Col<int> a_counts = arma::conv_to<arma::Col<int>>::from(b_rows) * a.n_rows;
    arma::Col<int> a_disp = arma::cumsum(a_counts) - a_counts;
    c = mat(m, k, arma::fill::zeros);
    mat a_buf;
    for (size_t pi = 0; pi < np; pi++)
    {
        a_buf.resize(b_rows(pi), b_rows(ip));
        MPI_Scatterv(a.memptr(), a_counts.mem, a_disp.mem, MPI_DOUBLE, a_buf.memptr(), a_buf.n_elem, MPI_DOUBLE, pi, MPI_COMM_WORLD);
        mat ci = a_buf * b;
        MPI_Reduce(ci.memptr(), c.memptr(), ci.n_elem, MPI_DOUBLE, MPI_SUM, pi, MPI_COMM_WORLD);
    }
}
