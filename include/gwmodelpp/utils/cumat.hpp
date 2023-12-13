#ifndef CUMAT_HPP
#define CUMAT_HPP

#include <exception>
#include <armadillo>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class cubase;
class cumat;
class custride;
template <class T>
class cuop_trans;
template<class T>
class cuview;

/**
 * @brief \~english Struct of range \~chinese 范围结构
 * 
 */
struct curange
{
    size_t start = 0;
    size_t end = 0;
};


/**
 * @brief \~english Base class of operator types \~chinese 运算符基类
 * 
 */
class cuop
{
public:
    enum class Op
    {
        Origin = CUBLAS_OP_N,
        Trans = CUBLAS_OP_T
    };
};

/**
 * @brief \~english Base class of types managing data \~chinese 数据类型的基类
 * 
 */
class cubase
{
public:
    static cublasHandle_t handle;   //!< Save handle for cublas
    static auto create_handle()
    {
        if (handle == nullptr) return cublasCreate(&handle);
        else return CUBLAS_STATUS_SUCCESS;
    }
    static auto destory_handle()
    {
        if (handle != nullptr)
        {
            auto error = cublasDestroy(handle);
            handle = nullptr;
            return error;
        }
        else return CUBLAS_STATUS_SUCCESS;
    }
    constexpr static const double alpha1 = 1.0;
    constexpr static const double beta0 = 0.0;
    constexpr static const double beta1 = 1.0;

    enum class Type
    {
        Base,
        Op,
        Mat,
        Stride,
        Batched
    };

    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Base;

    enum class Init
    {
        None,
        Zero
    };

public:

    /**
     * @brief \~english Construct a new cubase object. \~chinese 构造一个新的 cubase 对象。 
     * 
     */
    cubase() {}

    /**
     * @brief \~english Construct a new cubase object. \~chinese 构造一个新的 cubase 对象。 
     * 
     * @param bytes \~english Size in bytes \~chinese 字节数
     * @param init \~english How to initialise \~chinese 初始化方式
     */
    cubase(size_t bytes, Init init = Init::Zero)
    {
        switch (init)
        {
        case Init::Zero:
            mIsRelease = true;
            cudaMalloc(&dMem, bytes);
            cudaMemset(dMem, 0, bytes);
            break;
        default:
            break;
        }
    }

    /**
     * @brief \~english Destroy the cubase object. \~chinese 销毁 cubase 对象。 
     * 
     */
    virtual ~cubase()
    {
        if (mIsRelease && dMem)
        {
            cudaFree(dMem);
        }
        dMem = nullptr;
    }

    /**
     * @brief \~english Get size in bytes \~chinese 获取字节数
     * 
     * @return size_t \~english Size in bytes \~chinese 字节数
     */
    virtual size_t nbytes() const = 0;

    /**
     * @brief \~english Get pointer to data \~chinese 获取数据指针
     * 
     * @return double* \~english Pointer to data \~chinese 数据指针
     */
    double* dmem() const { return dMem; }

    /**
     * @brief \~english Get data in GPU \~chinese 提取 GPU 中的数据
     * 
     * @param dst \~english Pointer to where to store data \~chinese 指向要保存数据位置的指针
     */
    virtual void get(double* dst)
    {
        cudaMemcpy(dst, dMem, nbytes(), cudaMemcpyDeviceToHost);
    }

protected:
    bool mIsRelease = false;
    double* dMem = nullptr;
};

template<class T>
struct cutraits
{
    constexpr static cubase::Type type = T::type;
    constexpr static cuop::Op op = T::op;
};

template<class A, class B, cubase::Type TA = cutraits<A>::type, cubase::Type TB = cutraits<B>::type>
class cuop_matmul;

class cuop_inv;

class cuop_diagmul;

/**
 * @brief \~english Matrix. Elements are stored in GPU by column-major format. \~chinese 矩阵类。数据存储在 GPU 中，列主序格式。
 * 
 */
class cumat : public cubase
{
public:
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Mat;

public:

    /**
     * @brief \~english Construct a new cumat object. \~chinese 构造一个新的 cumat 对象。
     * 
     */
    cumat() {}

    /**
     * @brief \~english Construct a new cumat object. \~chinese 构造一个新的 cumat 对象。
     * 
     * @param rows \~english Number of rows \~chinese 行数
     * @param cols \~english Number of columns \~chinese 列数
     * @param init \~english How to initialise \~chinese 初始化方式
     */
    cumat(size_t rows, size_t cols, cubase::Init init = cubase::Init::Zero) : 
        cubase(rows * cols * sizeof(double), init),
        mRows(rows),
        mCols(cols)
    {
    }

    /**
     * @brief \~english Construct a new cumat object from armadillo matrix. \~chinese 从 armadillo 的矩阵类构造一个新的 cumat 对象。
     * 
     * @param src \~english Armadillo matrix \~chinese Armadillo 类库的的矩阵类
     */
    cumat(arma::mat src) : cumat(src.n_rows, src.n_cols)
    {
        cudaMemcpy(dMem, src.mem, nbytes(), cudaMemcpyHostToDevice);
    }

    /**
     * @brief \~english Copy construct a new cumat object. \~chinese 复制构造一个新的 cumat 对象。
     * 
     * @param mat \~english Source matrix \~chinese 源矩阵
     */
    cumat(const cumat& mat) : cumat(mat.mRows, mat.mCols)
    {
        cudaMemcpy(dMem, mat.dMem, nbytes(), cudaMemcpyDeviceToDevice);
    }

    /**
     * @brief \~english Move construct a new cumat object.
     * The source object will move the management of memory to the new object.
     * \~chinese 移动构造一个新的 cumat 对象。
     * 源对象将把内存管理移交给新对象。
     * 
     * @param mat \~english  \~chinese 
     */
    cumat(cumat&& mat) : mRows(mat.mRows), mCols(mat.mCols)
    {
        mIsRelease = true;
        dMem = mat.dMem;
        mat.mIsRelease = false;
    }

    template<class L, class R>
    cumat(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type>&& op);

    cumat(cuop_diagmul&& op);

    /**
     * @brief \~english Destroy the cumat object. \~chinese 销毁 cumat 对象。
     * 
     */
    virtual ~cumat()
    {
        mRows = 0;
        mCols = 0;
    }

    size_t nbytes() const override { return sizeof(double) * mRows * mCols; }

    /**
     * @brief \~english Transpose matrix. Do not do the calculation immediately, unless it is assigned to a new object.
     * \~chinese 转置矩阵。除非赋值到新的对象，否则并不执行计算。
     * 
     * @return const cuop_trans<cumat> \~english Object with transpose mark \~chinese 带有转置标记的对象
     */
    const cuop_trans<cumat> t() const;

    void resize(size_t rows, size_t cols)
    {
        if (dMem != nullptr && mIsRelease)
        {
            cudaFree(dMem);
        }
        cudaMalloc(&dMem, sizeof(double) * rows * cols);
    }

    cumat& operator=(const cumat& right)
    {
        resize(right.mRows, right.mCols);
        cudaMemcpy(dMem, right.dMem, nbytes(), cudaMemcpyDeviceToDevice);
        return *this;
    }

    cumat& operator=(cumat&& right)
    {
        mRows = right.mRows;
        mCols = right.mCols;
        dMem = right.dMem;
        mIsRelease = true;
        right.mIsRelease = false;
        return *this;
    }

    cumat& operator=(const cuop_trans<cumat>& right);

    template<class L, class R>
    cumat& operator=(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type>&& op);

    cumat& operator=(cuop_diagmul&& op);

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<cumat, R>(*this, right);
    }

    /**
     * @brief \~english Convert to object of custride type in which each column is a stride. \~chinese 转换为 custride 类型，每列作为一个 stride。
     * 
     * @return custride \~english Converted custride object \~chinese 转换后的 custride 对象
     */
    custride as_stride() const;

    /**
     * @brief \~english Multiply with a diagonal matrix. \~chinese 与对角矩阵相乘。
     * 
     * @param diag \~english Diagonal elements of the diagonal matrix \~chinese 对角矩阵的对角线元素
     * @return cumat \~english Result matrix \~chinese 结果矩阵
     */
    cuop_diagmul diagmul(const cumat& diag) const;

public:
    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }

protected:
    size_t mRows = 0;
    size_t mCols = 0;
};

void print(const cumat& mat);

/**
 * @brief \~english Strided matrix. \~chinese 条带矩阵。
 * 
 */
class custride: public cubase
{
public:
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Stride;

public:

    /**
     * @brief \~english Construct a new custride object. \~chinese 构造一个新的 custride 对象。
     * 
     */
    custride() {}

    /**
     * @brief \~english Construct a new custride object. \~chinese 构造一个新的 custride 对象。
     * 
     * @param rows \~english  \~chinese 
     * @param cols \~english  \~chinese 
     * @param strides \~english  \~chinese 
     * @param init \~english  \~chinese 
     */
    custride(size_t rows, size_t cols, size_t strides, cubase::Init init = cubase::Init::Zero) : 
        cubase(sizeof(double) * rows * cols * strides, init),
        mRows(rows),
        mCols(cols),
        mStrides(strides)
    {}

    /**
     * @brief \~english Construct a new custride object form armadillo cube. \~chinese 从 Armadillo 类库的 cube 对象构造一个新的 custride 对象。
     * 
     * @param src \~english  \~chinese 
     */
    explicit custride(const arma::cube& src):
        cubase(sizeof(double) * src.n_elem),
        mRows(src.n_rows),
        mCols(src.n_cols),
        mStrides(src.n_slices)
    {
        cudaMemcpy(dMem, src.mem, nbytes(), cudaMemcpyHostToDevice);
    }

    /**
     * @brief \~english Copy construct a new custride object. \~chinese 复制构造一个新的 custride 对象。
     * 
     * @param mat \~english  \~chinese 
     */
    custride(const custride& mat) : custride(mat.mRows, mat.mCols, mat.mStrides)
    {
        cudaMemcpy(dMem, mat.dMem, nbytes(), cudaMemcpyDeviceToDevice);
    }

    /**
     * @brief \~english Move construct a new custride object. \~chinese 移动构造一个新的 custride 对象。
     * 
     * @param mat \~english  \~chinese 
     */
    custride(custride&& mat) : mRows(mat.mRows), mCols(mat.mCols), mStrides(mat.mStrides)
    {
        mIsRelease = true;
        dMem = mat.dMem;
        mat.mIsRelease = false;
    }

    /**
     * @brief \~english Construct a new custride object from a cumat object. Each column will be a stride.
     * \~chinese 从 cumat 对象构造一个新的 custride 对象。每列是一个条带。
     * 
     * @param mat \~english  \~chinese 
     */
    explicit custride(const cumat& mat) : custride(mat.nrows(), 1, mat.ncols(), cubase::Init::None)
    {
        dMem = mat.dmem();
    }

    template<class L, class R>
    custride(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type>&& op);

    custride(cuop_inv&& op);

    /**
     * @brief \~english Destroy the custride object. \~chinese 销毁 custride 对象。
     * 
     */
    virtual ~custride()
    {
        mRows = 0;
        mCols = 0;
        mStrides = 0;
    }

    size_t nbytes() const override { return sizeof(double) * mRows * mCols * mStrides; }

    size_t nrows() const { return mRows; }
    size_t ncols() const { return mCols; }
    size_t nstrides() const { return mStrides; }
    size_t nstrideSize() const { return mRows * mCols; }
    size_t nstrideBytes() const { return mRows * mCols * sizeof(double); }

    /**
     * @brief \~english Get a stride at start \~chinese 获取 start 指定的条带
     * 
     * @param start \~english  \~chinese 
     * @return cuview<custride> \~english  \~chinese 
     */
    cuview<custride> strides(size_t start) const;

    /**
     * @brief \~english Get a range of strides \~chinese 获取条带范围
     * 
     * @param start \~english  \~chinese 
     * @param end \~english  \~chinese 
     * @return cuview<custride> \~english  \~chinese 
     */
    cuview<custride> strides(size_t start, size_t end) const;

    /**
     * @brief \~english Transpose matrix \~chinese 转置矩阵
     * 
     * @return const cuop_trans<custride> \~english  \~chinese 
     */
    const cuop_trans<custride> t() const;

    /**
     * @brief \~english Inverse matrix. Will calculate instantly.
     * \~chinese 求逆矩阵。该函数立即执行。
     * 
     * @param d_info \~english  \~chinese 
     * @return custride \~english  \~chinese 
     */
    cuop_inv inv(int* d_info) const;

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<custride, R>(*this, right);
    }

    template<class L, class R>
    custride& operator=(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type>&& op);

    custride& operator=(cuop_inv&& op);

protected:
    size_t mRows = 0;
    size_t mCols = 0;
    size_t mStrides = 0;
};

class cubatched
{
public:
    constexpr static cuop::Op op = cuop::Op::Origin;
    constexpr static cubase::Type type = cubase::Type::Batched;

public:
    cubatched() {}

    cubatched(size_t batch) : mBatch(batch)
    {
        cudaMalloc(&dArray, sizeof(double*) * mBatch);
        cudaMemset(dArray, 0, sizeof(double*) * mBatch);
    }

    cubatched(size_t batch, size_t rows, size_t cols, bool initialize = false) : cubatched(batch)
    {
        mRows = rows;
        mCols = cols;
        if (initialize)
        {
            mIsRelease = true;
            for (size_t i = 0; i < batch; i++)
            {
                cudaMalloc(dArray + i, sizeof(double) * rows * cols);
                cudaMemset(dArray[i], 0, sizeof(double) * rows * cols);
            }
        }
    }

    cubatched(const cubatched& right) : cubatched(right.mBatch, right.mRows, right.mCols, true)
    {
        for (size_t i = 0; i < mBatch; i++)
        {
            cudaMemcpy(dArray[i], right.dArray[i], sizeof(double) * mRows * mCols, cudaMemcpyDeviceToDevice);
        }
    }

    cubatched(cubatched&& right) : cubatched(right.mBatch, right.mRows, right.mCols, false)
    {
        dArray = right.dArray;
        right.mIsRelease = false;
    }

    cubatched(const double** p_mats, size_t batch, size_t rows, size_t cols) : cubatched(batch, rows, cols)
    {
        cudaMemcpy(dArray, p_mats, sizeof(double*) * batch, cudaMemcpyHostToDevice);        
    }

    cubatched(double* d_mat, size_t batch, size_t rows, size_t cols, size_t bias) : cubatched(batch, rows, cols)
    {
        double** p_mats = new double*[mBatch];
        for (size_t i = 0; i < mBatch; i++)
        {
            p_mats[i] = d_mat + i * bias;
        }
        cudaMemcpy(dArray, p_mats, sizeof(double*) * mBatch, cudaMemcpyHostToDevice);
        delete[] p_mats;
    }

    cubatched(const custride& stride) : cubatched(stride.dmem(), stride.nstrides(), stride.nrows(), stride.ncols(), stride.nstrideSize())
    {}

    cubatched(const cumat& mat) : cubatched(mat.dmem(), mat.ncols(), mat.nrows(), mat.nrows(), 1)
    {}

    cubatched(const cumat& mat, size_t batch) : cubatched(mat.dmem(), batch, 0, mat.nrows(), mat.ncols())
    {}

    ~cubatched()
    {
        if (mIsRelease && dArray)
        {
            for (size_t i = 0; i < mBatch; i++)
            {
                cudaFree(dArray[i]);
            }
        }
        cudaFree(dArray);
        dArray = nullptr;
    }

    double** darray() { return dArray; }
    size_t nrows() { return mRows; }
    size_t ncols() { return mCols; }
    size_t nbatch() { return mBatch; }

public:
    cubatched inv(int* dinfo)
    {
        cubatched res(mBatch, mRows, mCols, true);
        cublasDmatinvBatched(cubase::handle, mRows, dArray, mRows, res.darray(), mRows, dinfo, mBatch);
        return res;
    }

private:
    size_t mBatch = 0;
    double** dArray = nullptr;
    size_t mRows = 0;
    size_t mCols = 0;
    bool mIsRelease = false;
};

template<typename T>
class cuop_trans : public cuop
{
public:
    constexpr static cuop::Op op = cuop::Op::Trans;
    constexpr static cubase::Type type = cutraits<T>::type;

public:
    cuop_trans(const T& src): ori(src) {}

    cuop_trans(const cuop_trans& src) : ori(src.ori) {}

    double* dmem() const { return ori.dmem(); }
    size_t nrows() const { return ori.ncols(); }
    size_t ncols() const { return ori.nrows(); }

    template<class R>
    auto operator*(const R& right) const
    {
        return cuop_matmul<cuop_trans<T>, R, cutraits<T>::type, cutraits<R>::type>(*this, right);
    }

    const T& ori;
};

template<class T>
class cuview
{
public:
    constexpr static cubase::Type type = cutraits<T>::type;
    constexpr static cuop::Op op = cutraits<T>::op;

public:
    explicit cuview(T& src): mSrc(src) {}

protected:
    T& mSrc;
};

template<>
class cuview<custride>
{
public:
    constexpr static cubase::Type type = custride::type;
    constexpr static cuop::Op op = custride::op;

public:
    explicit cuview(const custride& src, curange strides): mSrc(src), mStrides(strides)
    {}
    
    size_t nrows() const { return mSrc.nrows(); }
    size_t ncols() const { return mSrc.ncols(); }
    size_t nstrides() const { return mStrides.end - mStrides.start; }
    size_t nstrideSize() const { return mSrc.nstrideSize(); }
    size_t nstrideBytes() const { return mSrc.nstrideBytes(); }
    size_t nbytes() const { return nstrides() * nstrideBytes(); }
    double* dmem() const { return mSrc.dmem() + mStrides.start * mSrc.nstrideSize(); } 

    void get(double* dst)
    {
        cudaMemcpy(dst, dmem(), nbytes(), cudaMemcpyDeviceToHost);
    }

    cuview<custride>& operator=(custride&& right)
    {
        cudaMemcpy(dmem(), right.dmem(), nbytes(), cudaMemcpyDeviceToDevice);
        return *this;
    }

    cuview<custride>& operator=(cumat&& right)
    {
        cudaMemcpy(dmem(), right.dmem(), nstrideBytes(), cudaMemcpyDeviceToDevice);
        return *this;
    }

    template<class L, class R>
    cuview<custride>& operator=(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type>&& op);

    cuview<custride>& operator=(cuop_diagmul&& op);

    cuop_trans<cuview<custride>> t()
    {
        return cuop_trans<cuview<custride>>(*this);
    }

protected:
    const custride& mSrc;
    const curange mStrides;
};

/**
 * @brief \~english Operator of matrix multiply \~chinese 矩阵乘法运算符
 * 
 * @tparam A \~english Type of left operands \~chinese 左操作数的类型
 * @tparam B \~english Type of right operands \~chinese 右操作数的类型
 * @tparam TA \~english Traits of left operands \~chinese 左操作数的本质
 * @tparam TB \~english Traits of right operands \~chinese 右操作数的本质
 */
template<class A, class B, cubase::Type TA, cubase::Type TB>
class cuop_matmul
{
public:
    cuop_matmul(const A& left, const B& right): a(left), b(right) {}

    template<class T>
    int getStrides(const T& m) const
    {
        return 1;
    }

    template<class T>
    int getStrideSize(const T& m) const
    {
        return cutraits<T>::type == cubase::Type::Stride ? m.nrows() * m.ncols() : 0;
    }

    int getStrides(const custride& m) const
    {
        return m.nstrides();
    }

    int getStrides(const cuop_trans<custride>& m) const
    {
        return m.ori.nstrides();
    }

    int getStrides(const cuview<custride>& m) const
    {
        return m.nstrides();
    }

    int getStrides(const cuop_trans<cuview<custride>>& m) const
    {
        return m.ori.nstrides();
    }

    size_t nrows() const { return a.nrows(); }

    size_t ncols() const { return b.ncols(); }

    size_t nstrides() const { return cutraits<A>::type == cubase::Type::Stride ? getStrides(a) : (cutraits<B>::type == cubase::Type::Stride ? getStrides(b) : 1); }

    template<class C>
    void eval(C& c)
    {
        size_t m = a.nrows(), k = a.ncols(), n = b.ncols();
        int lda = cutraits<A>::op == cuop::Op::Origin ? a.nrows() : a.ncols();
        int ldb = cutraits<B>::op == cuop::Op::Origin ? b.nrows() : b.ncols();
        auto opa = cutraits<A>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto opb = cutraits<B>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        size_t strideSizeA = getStrideSize(a);
        size_t strideSizeB = getStrideSize(b);
        size_t strideC = getStrides(c);
        int strideSizeC = getStrideSize(c);
        cublasStatus_t error = cublasDgemmStridedBatched(
            cubase::handle, opa, opb,
            m, n, k, &cubase::alpha1,
            a.dmem(), lda, strideSizeA,
            b.dmem(), ldb, strideSizeB,
            &cubase::beta0, c.dmem(), m, strideSizeC, strideC
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw cublasGetStatusString(error);
    }

private:
    const A& a;
    const B& b;
};

/**
 * @brief \~english Operator of matrix multiply specificated for matrix-matrix multiply \~chinese 矩阵乘法运算符，为矩阵与矩阵乘法特化
 * 
 * @tparam A \~english Type of left operands \~chinese 左操作数的类型
 * @tparam B \~english Type of right operands \~chinese 右操作数的类型
 */
template<class A, class B>
class cuop_matmul<A, B, cubase::Type::Mat, cubase::Type::Mat>
{
public:

    cuop_matmul(const A& left, const B& right): a(left), b(right) {}

    size_t nrows() const { return a.nrows(); }

    size_t ncols() const { return b.ncols(); }

    template<class C>
    void eval(C& c)
    {
        size_t m = a.nrows(), k = a.ncols(), n = b.ncols();
        int lda = cutraits<A>::op == cuop::Op::Origin ? a.nrows() : a.ncols();
        int ldb = cutraits<B>::op == cuop::Op::Origin ? b.nrows() : b.ncols();
        auto opa = cutraits<A>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto opb = cutraits<B>::op == cuop::Op::Origin ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasStatus_t error = cublasDgemm(
            cubase::handle, opa, opb,
            m, n, k, &cubase::alpha1,
            a.dmem(), lda,
            b.dmem(), ldb,
            &cubase::beta0, c.dmem(), m
        );
        if (error != CUBLAS_STATUS_SUCCESS) throw cublasGetStatusString(error);
    }

private:
    const A& a;
    const B& b;
};

class cuop_inv
{
public:
    cuop_inv(const custride& left, int* info): a(left), d_info(info) {};

    size_t nrows() const { return a.nrows(); }
    size_t ncols() const { return a.nrows(); }
    size_t nstrides() const { return a.nstrides(); }

    void eval(custride& c)
    {
        size_t n = a.nrows();
        cubatched b_array(a), b_inv(c);
        cublasDmatinvBatched(cubase::handle, a.nrows(), b_array.darray(), n, b_inv.darray(), n, d_info, b_array.nbatch());
    }

private:
    const custride& a;
    int* d_info;
};

class cuop_diagmul
{
public:
    cuop_diagmul(const cumat& left, const cumat& right): a(left), b(right) {};

    size_t nrows() const { return a.nrows(); }
    size_t ncols() const { return a.ncols(); }

    template<class C>
    void eval(C& c)
    {
        cublasDdgmm(
            cubase::handle, CUBLAS_SIDE_RIGHT, a.nrows(), a.ncols(), 
            a.dmem(), a.nrows(), 
            b.dmem(), 1, 
            c.dmem(), c.nrows()
        );
    }

private:
    const cumat& a;
    const cumat& b;
};

template <class L, class R>
inline cumat::cumat(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type> &&op): cumat(op.nrows(), op.ncols())
{
    op.eval(*this);
}

template <class L, class R>
inline cumat &cumat::operator=(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type> &&op)
{
    op.eval(*this);
    return *this;
}

template <class L, class R>
inline custride::custride(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type> &&op): custride(op.nrows(), op.ncols(), op.nstrides())
{
    op.eval(*this);
}

template <class L, class R>
inline custride &custride::operator=(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type> &&op)
{
    op.eval(*this);
    return *this;
}

template <class L, class R>
inline cuview<custride> &cuview<custride>::operator=(cuop_matmul<L, R, cutraits<L>::type, cutraits<R>::type> &&op)
{
    op.eval(*this);
    return *this;
}

inline cuview<custride> &cuview<custride>::operator=(cuop_diagmul &&op)
{
    op.eval(*this);
    return *this;
}

#endif  // CUMAT_HPP
