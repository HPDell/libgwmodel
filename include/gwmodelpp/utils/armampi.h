#include <armadillo>

void mat_mul_mpi(arma::mat& a, arma::mat& b, arma::mat& c, const int ip, const int np, const size_t range);
