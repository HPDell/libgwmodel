#ifndef IPARALLELIZABLE_H
#define IPARALLELIZABLE_H

namespace gwm
{

/**
 * \~english
 * @brief Parallel types.
 * 
 * \~chinese
 * @brief 并行方法类型。
 * 
 */
enum ParallelType
{
    SerialOnly = 1 << 0,    //!< \~english Use no parallel methods. \~chinese 不并行。
    OpenMP = 1 << 1,        //!< \~english Use multithread methods. \~chinese 多线程并行。
    CUDA = 1 << 2,          //!< \~english Use CUDA accelerated methods. \~chinese CUDA加速。
    MPI = 1 << 3
};

/**
 * \~english
 * @brief Interface for parallelizable algorithm.
 * 
 * \~chinese
 * @brief 并行算法接口。
 */
struct IParallelizable
{
    /**
     * \~english
     * @brief Return the parallel ability of this algorithm.
     * 
     * @return Bitwise OR of aviliable parallel types of this algorithm.
     * 
     * \~chinese
     * @brief 返回该算法的并行能力。
     * 
     * @return 当前算法并行能力的按位或运算结果。
     * 
     */
    virtual int parallelAbility() const = 0;

    /**
     * \~english
     * @brief Return the parallel type of this algorithm.
     * 
     * @return Parallel type of this algorithm 
     * 
     * \~chinese
     * @brief 返回当前算法的并行类型。
     * 
     * @return 当前算法的并行类型。 
     * 
     */
    virtual ParallelType parallelType() const = 0;

    /**
     * \~english
     * @brief Set the parallel type of this algorithm.
     * 
     * @param type Parallel type of this algorithm.
     * 
     * \~chinese
     * @brief 设置当前算法的并行类型。
     * 
     * @param type 当前算法的并行类型。
     * 
     */
    virtual void setParallelType(const ParallelType& type) = 0;
};

/**
 * \~english
 * @brief Interface for parallelizable algorithm implemented by OpenMP.
 * 
 * \~chinese
 * @brief 可 OpenMP 并行的算法接口。
 * 
 */
struct IParallelOpenmpEnabled
{
    /**
     * \~english
     * @brief Set the thread numbers while paralleling.
     * 
     * @param threadNum Number of threads.
     * 
     * \~chinese
     * @brief 设置并行线程数。
     * 
     * @param threadNum 并行线程数。
     */
    virtual void setOmpThreadNum(const int threadNum) = 0;
};

/**
 * \~english
 * @brief Interface for parallelizable algorithm implemented by CUDA.
 * 
 * \~english
 * @brief 可 CUDA 加速的算法接口。
 * 
 */
struct IParallelCudaEnabled
{
    /**
     * \~english
     * @brief Set ID of used GPU while paralleling. 
     * 
     * @param gpuId ID of used GPU. Start from 0.
     * 
     * \~chinese
     * @brief 设置并行化使用的 GPU 的 ID。
     * 
     * @param gpuId 并行化使用的 GPU 的 ID。 从 0 开始。
     * 
     */
    virtual void setGPUId(const int gpuId) = 0;

    /**
     * \~english
     * @brief Set the group size while paralleling.
     * 
     * @param size Group size \f$g\f$ while paralleling.
     * Usually this is up to memory size \f$m\f$ (B) of GPU with ID gpuId.
     * If there are \f$n\f$ samples and \f$k\f$ variables, \f[ k * n * g * 8 < m \f]
     * For most GPU, 64 is OK.
     * 
     * \~chinese
     * @brief Set the group size while paralleling.
     * 
     * @param size 并行化时的组长度 \f$g\f$。
     * 通常取决于 gpuID 对应显卡的显存 \f$m\f$ (B)。
     * 如果有 \f$n\f$ 样本和 \f$k\f$ 变量， 要求\f[ k * n * g * 8 < m \f] 。
     * 对于大多数 GPU 可选择值 64。
     * 
     */
    virtual void setGroupSize(const double size) = 0;
    
};

struct IParallelMpiEnabled
{
    virtual int workerId() = 0;
    virtual void setWorkerId(int id) = 0;
    virtual void setWorkerNum(int size) = 0;
};

#define GWM_MPI_MASTER_BEGIN if (workerId() == 0) {
#define GWM_MPI_MASTER_END }
#define GWM_MPI_WORKER_BEGIN if (workerId() != 0) {
#define GWM_MPI_WORKER_END }
#define GWM_MPI_MASTER_WORKER_SWITCH } else {

}

#endif  // IPARALLELIZABLE_H