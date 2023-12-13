#include "gwmodelpp/spatialweight/CRSDistance.h"
#include <assert.h>
#include <exception>

#ifdef ENABLE_CUDA
#include "CudaUtils.h"
#endif // ENABLE_CUDA

using namespace std;
using namespace arma;
using namespace gwm;

#define POWDI(x, i) pow(x, i)

double CRSDistance::SpGcdist(double lon1, double lon2, double lat1, double lat2)
{

    double F, G, L, sinG2, cosG2, sinF2, cosF2, sinL2, cosL2, S, C;
    double w, R, a, f, D, H1, H2;
    double lat1R, lat2R, lon1R, lon2R, DE2RA;

    DE2RA = M_PI / 180;
    a = 6378.137;            /* WGS-84 equatorial radius in km */
    f = 1.0 / 298.257223563; /* WGS-84 ellipsoid flattening factor */

    if (fabs(lat1 - lat2) < DOUBLE_EPS)
    {
        if (fabs(lon1 - lon2) < DOUBLE_EPS)
        {
            return 0.0;
            /* Wouter Buytaert bug caught 100211 */
        }
        else if (fabs((fabs(lon1) + fabs(lon2)) - 360.0) < DOUBLE_EPS)
        {
            return 0.0;
        }
    }
    lat1R = lat1 * DE2RA;
    lat2R = lat2 * DE2RA;
    lon1R = lon1 * DE2RA;
    lon2R = lon2 * DE2RA;

    F = (lat1R + lat2R) / 2.0;
    G = (lat1R - lat2R) / 2.0;
    L = (lon1R - lon2R) / 2.0;

    /*
    printf("%g %g %g %g; %g %g %g\n",  *lon1, *lon2, *lat1, *lat2, F, G, L);
    */

    sinG2 = POWDI(sin(G), 2);
    cosG2 = POWDI(cos(G), 2);
    sinF2 = POWDI(sin(F), 2);
    cosF2 = POWDI(cos(F), 2);
    sinL2 = POWDI(sin(L), 2);
    cosL2 = POWDI(cos(L), 2);

    S = sinG2 * cosL2 + cosF2 * sinL2;
    C = cosG2 * cosL2 + sinF2 * sinL2;

    w = atan(sqrt(S / C));
    R = sqrt(S * C) / w;

    D = 2 * w * a;
    H1 = (3 * R - 1) / (2 * C);
    H2 = (3 * R + 1) / (2 * S);

    return D * (1 + f * H1 * sinF2 * cosG2 - f * H2 * cosF2 * sinG2);
}

vec CRSDistance::SpatialDistance(const rowvec &out_loc, const mat &in_locs)
{
    uword N = in_locs.n_rows;
    vec dists(N, fill::zeros);
    double uout = out_loc(0), vout = out_loc(1);
    for (uword j = 0; j < N; j++)
    {
        dists(j) = SpGcdist(in_locs(j, 0), uout, in_locs(j, 1), vout);
    }
    return dists;
}

CRSDistance::CRSDistance(const CRSDistance &distance) : Distance(distance)
{
    setGeographic(distance.mGeographic);
    if (distance.mParameter)
    {
        mat fp = distance.mParameter->focusPoints;
        mat dp = distance.mParameter->dataPoints;
        mParameter = make_unique<Parameter>(fp, dp);
#ifdef ENABLE_CUDA
        mUseCuda = distance.mUseCuda;
        mCudaPrepared = false;
        if (distance.mUseCuda)
        {
            prepareCuda(distance.mGpuID);
        }
#endif // ENABLE_CUDA
    }
}

void CRSDistance::makeParameter(initializer_list<DistParamVariant> plist)
{
    if (plist.size() == 2)
    {
        const mat& fp = get<mat>(*(plist.begin()));
        const mat& dp = get<mat>(*(plist.begin() + 1));
        if (fp.n_cols == 2 && dp.n_cols == 2)
            mParameter = make_unique<Parameter>(fp, dp);
        else 
        {
            mParameter.reset(nullptr);
            throw std::runtime_error("The dimension of data points or focus points is not 2."); 
        }
    }
    else
    {
        mParameter.reset(nullptr);
        throw std::runtime_error("The number of parameters must be 2.");
    }
}

vec CRSDistance::distance(uword focus)
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (focus < mParameter->total)
    {
        return mCalculator(mParameter->focusPoints.row(focus), mParameter->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}

double CRSDistance::maxDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double maxD = 0.0;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = max(mCalculator(mParameter->focusPoints.row(i), mParameter->dataPoints));
        maxD = d > maxD ? d : maxD;
    }
    return maxD;
}

double CRSDistance::minDistance()
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    double minD = DBL_MAX;
    for (uword i = 0; i < mParameter->total; i++)
    {
        double d = min(mCalculator(mParameter->focusPoints.row(i), mParameter->dataPoints));
        minD = d < minD ? d : minD;
    }
    return minD;
}

#ifdef ENABLE_CUDA
cudaError_t CRSDistance::prepareCuda(size_t gpuId)
{
    checkCudaErrors(Distance::prepareCuda(gpuId));
    if (!mCudaPrepared)
    {
        mCudaDp = std::move(cumat(mParameter->dataPoints.t()));
        mCudaFp = std::move(cumat(mParameter->focusPoints.t()));
        mCudaPrepared = true;
    }
    return cudaSuccess;
}

cudaError_t CRSDistance::distance(uword focus, double *d_dists, size_t *elems)
{
    if (mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");
    if (!mCudaPrepared) throw std::logic_error("[CRSDistance] Cuda has not been prepared.");
    if (mCudaDp.dmem() == nullptr || mCudaFp.dmem() == nullptr || mCudaThreads == 0) throw std::logic_error("Cuda has not been correctly prepared.");
    if (focus < mParameter->total)
    {
        size_t fbias = focus * mParameter->focusPoints.n_cols;
        *elems = mParameter->total;
        return mCalculatorCuda(mCudaDp.dmem(), mCudaFp.dmem() + fbias, mParameter->total, mCudaThreads, d_dists);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}
#endif // ENABLE_CUDA
