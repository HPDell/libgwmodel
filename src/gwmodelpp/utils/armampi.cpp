#include "gwmodelpp/utils/armampi.h"
#include <mpi.h>
#include <memory>

using namespace std;
using namespace arma;

void mat_mul_mpi(mat& a, mat& b, mat& c, const int ip, const int np, const size_t range)
{
    arma::uvec a_rows(np, fill::zeros);
    MPI_Allgather(&b.n_rows, 1, MPI_UNSIGNED_LONG_LONG, a_rows.memptr(), 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    arma::uvec a_disp = cumsum(a_rows) - a_rows;
    auto m = a.n_rows, n = b.n_cols, k = a.n_cols;
    c = mat(m, k, fill::zeros);
    mat a_buf(range, n, fill::zeros);
    for (size_t pi = 0; pi < np; pi++)
    {
        if (pi == ip)
        {
            a_buf = a;
        }
        else
        {
            a_buf.resize(a_rows(pi), n);
        }
        MPI_Bcast(a_buf.memptr(), a_buf.n_elem, MPI_DOUBLE, pi, MPI_COMM_WORLD);
        mat ci = a_buf.cols(a_disp(ip), a_disp(ip) + a_rows(ip) - 1) * b;
        MPI_Reduce(ci.memptr(), c.memptr(), ci.n_elem, MPI_DOUBLE, MPI_SUM, pi, MPI_COMM_WORLD);
    }
}
