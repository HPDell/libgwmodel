#include "gwmodelpp/utils/armampi.h"
#include <mpi.h>
#include <memory>

using namespace std;
using namespace arma;

void mat_mul_mpi(mat& a, mat& b, mat& c, const int ip, const int np, const size_t range)
{
    auto m = a.n_rows, n = b.n_cols, k = a.n_cols;
    mat at = a.t();
    c = mat(n, m, fill::zeros);
    unique_ptr<double[], std::default_delete<double[]>> a_buf(new double[range]);
    unique_ptr<double[], std::default_delete<double[]>> c_buf(new double[n]);
    for (size_t p = 0; p < np; p++)
    {
        arma::uvec counts(np, fill::zeros);
        MPI_Gather(&a.n_rows, 1, MPI_UNSIGNED_LONG_LONG, counts.memptr(), counts.n_elem, MPI_UNSIGNED_LONG_LONG, p, MPI_COMM_WORLD);
        arma::Col<int> icounts = arma::conv_to<arma::Col<int>>::from(counts);
        arma::Col<int> displs = cumsum(icounts) - icounts;
        for (size_t i = 0; i < at.n_cols; i++)
        {
            vec ai = at.col(i);
            MPI_Scatterv(ai.memptr(), icounts.mem, displs.mem, MPI_DOUBLE, a_buf.get(), range, MPI_DOUBLE, p, MPI_COMM_WORLD);
            arma::rowvec ai_row(a_buf.get(), b.n_rows);
            // ci[1 x n] = ai[1 x r] * b[r x n]
            mat ci = ai.t() * b;
            vec c_reduce(n);
            MPI_Reduce(ci.memptr(), c_reduce.memptr(), n, MPI_DOUBLE, MPI_SUM, p, MPI_COMM_WORLD);
            c.col(i) = c_reduce;
        }
    }
    c = c.t();
}
