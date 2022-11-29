#include "gwmodelpp/spatialweight/CGwmCRSDistance.h"
#include <assert.h>

#include <exception>

#define POWDI(x, i) pow(x, i)

double CGwmCRSDistance::SpGcdist(double lon1, double lon2, double lat1, double lat2)
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

vec CGwmCRSDistance::SpatialDistance(const rowvec &out_loc, const mat &in_locs)
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

CGwmCRSDistance::CGwmCRSDistance(const CGwmCRSDistance &distance) : CGwmDistance(distance)
{
    mGeographic = distance.mGeographic;
    if (distance.mParameter)
    {
        mat fp = distance.mParameter->focusPoints;
        mat dp = distance.mParameter->dataPoints;
        mParameter = make_unique<Parameter>(fp, dp);
    }
}

void CGwmCRSDistance::makeParameter(initializer_list<DistParamVariant> plist)
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

vec CGwmCRSDistance::distance(uword focus)
{
    if(mParameter == nullptr) throw std::runtime_error("Parameter is nullptr.");

    if (focus < mParameter->total)
    {
        return mCalculator(mParameter->focusPoints.row(focus), mParameter->dataPoints);
    }
    else throw std::runtime_error("Target is out of bounds of data points.");
}

double CGwmCRSDistance::maxDistance()
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

double CGwmCRSDistance::minDistance()
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
