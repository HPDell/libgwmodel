#ifndef CRSDISTANCE_H
#define CRSDISTANCE_H

#include "Distance.h"

#ifdef ENABLE_CUDA
#include "gwmodelpp/utils/cumat.hpp"
#include "gwmodelpp/spatialweight/cuda/CRSDistanceKernel.h"
#include "gwmodelpp/spatialweight/cuda/ISpatialCudaEnabled.h"
#endif // ENABLE_CUDA

namespace gwm
{

/**
 * @brief \~english Class for calculating spatial distance according to coordinate reference system. \~chinese 
 */
class CRSDistance : public Distance
{
public:

    /**
     * @brief \~english Struct of parameters used in spatial distance calculating according to coordinate reference system. 
     * \~chinese 距离计算用的参数。
     */
    struct Parameter : public Distance::Parameter
    {
        /**
         * @brief \~english Matrix of focus points' coordinates.
         * The shape of it must be \f$n \times 2\f$ and the first column is longitudes or \f$x\f$-coordinate,
         * the second column is latitudes or \f$y\f$-coordinate.
         * \~chinese 目标点坐标矩阵。
         * 其形状必须是 \f$n \times 2\f$ 且其第一列是经度或 \f$x\f$ 坐标，
         * 第二列是纬度或 \f$y\f$ 坐标。
         */
        arma::mat focusPoints;

        /**
         * @brief \~english Matrix of data points' coordinates.
         * The shape of it must be \f$n \times 2\f$ and the first column is longitudes or \f$x\f$-coordinate,
         * the second column is latitudes or \f$y\f$-coordinate.
         * \~chinese 数据点坐标矩阵。
         * 其形状必须是 \f$n \times 2\f$ 且其第一列是经度或 \f$x\f$ 坐标，
         * 第二列是纬度或 \f$y\f$ 坐标。
         */
        arma::mat dataPoints;

        /**
         * @brief \~english Construct a new CRSDistanceParameter object. \~chinese 构造一个新的 CRSDistanceParameter 对象。
         * 
         * @param fp \~english Reference to focus points \~chinese 目标点坐标引用
         * @param dp \~english Reference to data points \~chinese 数据点坐标引用
         */
        Parameter(const arma::mat& fp, const arma::mat& dp) : Distance::Parameter()
            , focusPoints(fp)
            , dataPoints(dp)
        {
            total = fp.n_rows;
        }
    };

public:

    /**
     * @brief \~english Calculate spatial distance for points with geographical coordinate reference system. \~chinese 计算地理坐标系下的空间距离。
     * 
     * @param out_loc \~english Row-wise vector of focus point' coordinate.
     * The shape of it must be 1x2 and the first column is longitudes, the second column is latitudes.
     * \~chinese 目标点坐标行向量。
     * 其形状必须是 \f$n \times 2\f$ 且其第一列是经度或 \f$x\f$ 坐标，
     * 第二列是纬度或 \f$y\f$ 坐标。
     * @param in_locs \~english Matrix of data points' coordinates.
     * The shape of it must be nx2 and the first column is longitudes, the second column is latitudes.
     * \~chinese 数据点坐标矩阵。
     * 其形状必须是 \f$n \times 2\f$ 且其第一列是经度或 \f$x\f$ 坐标，
     * 第二列是纬度或 \f$y\f$ 坐标。
     * @return arma::vec \~english Distance vector for out_loc \~chinese 为 out_loc 计算得到的距离向量
     */
    static arma::vec SpatialDistance(const arma::rowvec& out_loc, const arma::mat& in_locs);

    /**
     * @brief \~english Calculate euclidean distance for points with projected coordinate reference system. \~chinese 计算投影坐标系下的空间距离。
     * 
     * @param out_loc \~english Row-wise vector of focus point' coordinate.
     * The shape of it must be 1x2 and the first column is longitudes, the second column is latitudes.
     * \~chinese 目标点坐标行向量。
     * 其形状必须是 \f$n \times 2\f$ 且其第一列是经度或 \f$x\f$ 坐标，
     * 第二列是纬度或 \f$y\f$ 坐标。
     * @param in_locs \~english Matrix of data points' coordinates.
     * The shape of it must be nx2 and the first column is longitudes, the second column is latitudes.
     * \~chinese 数据点坐标矩阵。
     * 其形状必须是 \f$n \times 2\f$ 且其第一列是经度或 \f$x\f$ 坐标，
     * 第二列是纬度或 \f$y\f$ 坐标。
     * @return arma::vec \~english Distance vector for out_loc \~chinese 为 out_loc 计算得到的距离向量
     */
    static arma::vec EuclideanDistance(const arma::rowvec& out_loc, const arma::mat& in_locs)
    {
        arma::mat diff = (in_locs.each_row() - out_loc);
        return sqrt(sum(diff % diff, 1));
    }

    /**
     * @brief \~english Calculate spatial distance for two points with geographical coordinate reference system.
     * \~chinese 计算两个点之间的地理参考系距离。
     * 
     * @param lon1 \~english Longitude of point 1 \~chinese 第一个点的经度
     * @param lon2 \~english Longitude of point 2 \~chinese 第二个点的经度
     * @param lat1 \~english Latitude of point 1 \~chinese 第一个点的纬度
     * @param lat2 \~english Latitude of point 2 \~chinese 第二个点的纬度
     * @return double \~english Spatial distance for point 1 and point 2 \~chinese 两个点之间的空间距离
     */
    static double SpGcdist(double lon1, double lon2, double lat1, double lat2);

private:
    typedef arma::vec (*CalculatorType)(const arma::rowvec&, const arma::mat&);
#ifdef ENABLE_CUDA
    typedef cudaError_t (*CalculatorCudaType)(const double*, const double*, size_t, size_t, double*);
#endif

public:

    /**
     * @brief \~english Construct a new CRSDistance object. \~chinese 构造一个新的 CRSDistance 对象。
     * 
     */
    CRSDistance() : mGeographic(false), mParameter(nullptr) {}

    /**
     * @brief \~english Construct a new CRSDistance object. \~chinese 构造一个新的 CRSDistance 对象。
     * 
     * @param isGeographic \~english Whether the coordinate reference system is geographical \~chinese 坐标参考是是否是地理坐标系
     */
    explicit CRSDistance(bool isGeographic): mGeographic(isGeographic), mParameter(nullptr)
    {
        mCalculator = mGeographic ? &SpatialDistance : &EuclideanDistance;
#ifdef ENABLE_CUDA
        mCalculatorCuda = mGeographic ? &sp_dist_cuda : eu_dist_cuda;
#endif
    }

    /**
     * @brief \~english Copy construct a new CRSDistance object. \~chinese 拷贝构造一个新的 CRSDistance 对象。
     * 
     * @param distance \~english Reference to object for copying \~chinese 要拷贝的对象的引用
     */
    CRSDistance(const CRSDistance& distance);

    virtual ~CRSDistance()
    {
    }

    virtual Distance * clone() const override
    {
        return new CRSDistance(*this);
    }

    DistanceType type() override { return DistanceType::CRSDistance; }

    /**
     * @brief \~english Get whether the coordinates reference system is geographical. \~chinese 获取参考系是否是地理坐标系。
     * 
     * @return true \~english if the coordinate reference system is geographical \~chinese 如果坐标系是地理的
     * @return false \~english if the coordinate reference system is not geographical \~chinese 如果坐标系不是地理的
     */
    bool geographic() const
    {
        return mGeographic;
    }

    /**
     * @brief \~english Set whether the coordinates reference system is geographical. \~chinese 设置参考系是否是地理坐标系。
     * 
     * @param geographic \~english Whether the coordinate reference system is geographical \~chinese 参考系是否是地理坐标系
     */
    void setGeographic(bool geographic)
    {
        mGeographic = geographic;
        mCalculator = mGeographic ? &SpatialDistance : &EuclideanDistance;
#ifdef ENABLE_CUDA
        mCalculatorCuda = mGeographic ? &sp_dist_cuda : eu_dist_cuda;
#endif
    }

public:

    /**
     * @brief \~english Create Parameter for Caclulating CRS Distance. \~chinese 创建计算坐标系距离的参数。
     * 
     * @param plist \~english A list of parameters containing 2 items:
     *  - `arma::mat` focus points
     *  - `arma::mat` data points
     *  .
     * \~chinese 包含如下2项的参数列表：
     *  - `arma::mat` 目标点
     *  - `arma::mat` 数据点
     *  .
     * 
     */
    virtual void makeParameter(std::initializer_list<DistParamVariant> plist) override;

    virtual arma::vec distance(arma::uword focus) override;
    virtual double maxDistance() override;
    virtual double minDistance() override;

#ifdef ENABLE_CUDA
    virtual cudaError_t prepareCuda(size_t gpuId) override;

    virtual cudaError_t distance(arma::uword focus, double* d_dists, size_t* elems) override;
#endif

protected:
    bool mGeographic;  //!< \~english Whether the CRS is geographic \~chinese 坐标系是否是地理坐标系
    std::unique_ptr<Parameter> mParameter;  //!< \~english Parameters \~chinese 计算参数

private:
    CalculatorType mCalculator = &EuclideanDistance;  //!< \~english Calculator \~chinese 距离计算方法

#ifdef ENABLE_CUDA
    cumat mCudaDp;    //!< \~english Device pointer to data points \~chinese 指向数据点的设备指针
    cumat mCudaFp;    //!< \~english Device pointer to focus points \~chinese 指向关注点的设备指针
    CalculatorCudaType mCalculatorCuda = &eu_dist_cuda;  //!< \~english CUDA based Calculator \~chinese 基于 CUDA 的距离计算方法
#endif

};

}

#endif // CRSDISTANCE_H
