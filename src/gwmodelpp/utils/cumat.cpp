#include "cumat.hpp"

cublasHandle_t cubase::handle = nullptr;

cumat::cumat(cuop_diagmul &&op): cumat(op.nrows(), op.ncols())
{
    op.eval(*this);
}

const cuop_trans<cumat> cumat::t() const
{
    return cuop_trans<cumat>(*this);
}

custride cumat::as_stride() const
{
    return custride(*this);
}

cuop_diagmul cumat::diagmul(const cumat& diag) const
{
    return cuop_diagmul(*this, diag);
}

custride::custride(cuop_inv &&op): custride(op.nrows(), op.nrows(), op.nstrides())
{
    op.eval(*this);
}

cuview<custride> custride::strides(size_t start) const
{
    curange range { start, start + 1 };
    cuview<custride> view(*this, range);
    return view;
}

cuview<custride> custride::strides(size_t start, size_t end) const
{
    curange range { start, end };
    cuview<custride> view(*this, range);
    return view;
}

const cuop_trans<custride> custride::t() const
{
    return cuop_trans<custride>(*this);
}

cuop_inv custride::inv(int* d_info) const
{
    return cuop_inv(*this, d_info);
}

custride &custride::operator=(cuop_inv &&op)
{
    op.eval(*this);
    return *this;
}

cumat& cumat::operator=(const cuop_trans<cumat>& right)
{
    mRows = right.ori.ncols(), mCols = right.ori.nrows();
    if (dMem) cudaFree(dMem);
    cudaMalloc(&dMem, nbytes());
    auto error = cublasDgeam(
        cubase::handle, CUBLAS_OP_T, CUBLAS_OP_T, mRows, mCols, 
        &cubase::alpha1, right.ori.dmem(), right.ori.nrows(), 
        &cubase::beta0, right.ori.dmem(), right.ori.nrows(),
        dMem, mRows
    );
    if (error != CUBLAS_STATUS_SUCCESS) throw error;
    return *this;
}

cumat &cumat::operator=(cuop_diagmul &&op)
{
    op.eval(*this);
    return *this;
}

void print(cumat &mat)
{
    arma::mat am(mat.nrows(), mat.ncols());
    mat.get(am.memptr());
    am.brief_print();
}
