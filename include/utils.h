// utils.h  
#ifndef UTILS_H  
#define UTILS_H  

#include <cublas_v2.h>  
#include <cuda_runtime.h>  
#include <cuda_fp16.h>  
#include <cusolverDn.h>  
#include <vector>  
#include <iostream>  
#include <fstream>  

extern bool check;


// 定义错误检查宏  
#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl;                   \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }  

#define CHECK_CUBLAS(call)                                                       \
    {                                                                            \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }  

#define CHECK_CUSOLVER(call)                                                     \
    {                                                                            \
        cusolverStatus_t status = call;                                          \
        if (status != CUSOLVER_STATUS_SUCCESS) {                                 \
            std::cerr << "cuSOLVER error in " << __FILE__ << ":" << __LINE__     \
                      << std::endl;                                              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

/**  
 * @brief Namespace containing utility functions for CUDA operations.  
 */  
namespace Utils {  

/**  
 * @brief Computes the Euclidean norm (L2 norm) of a single-precision matrix.  
 *  
 * @param m    Number of rows in the matrix.  
 * @param n    Number of columns in the matrix.  
 * @param dA   Pointer to the device-side single-precision matrix.  
 * @return     The computed L2 norm as a float.  
 */  
float Snorm(long int m, long int n, float *dA);  

/**  
 * @brief Computes the Euclidean norm (L2 norm) of a double-precision matrix.  
 *  
 * @param m    Number of rows in the matrix.  
 * @param n    Number of columns in the matrix.  
 * @param dA   Pointer to the device-side double-precision matrix.  
 * @return     The computed L2 norm as a double.  
 */  
double dnorm(long int m, long int n, double *dA);  

/**  
 * @brief Computes the Euclidean norm (L2 norm) of a single-precision vector.  
 *  
 * @param m    Number of elements in the vector.  
 * @param x    Pointer to the device-side single-precision vector.  
 * @return     The computed L2 norm as a float.  
 */  
float SVnorm(int m, float *x);  

/**  
 * @brief Computes the Euclidean norm (L2 norm) of a double-precision vector.  
 *  
 * @param m    Number of elements in the vector.  
 * @param x    Pointer to the device-side double-precision vector.  
 * @return     The computed L2 norm as a double.  
 */  
double dVnorm(long int m, double *x);  

/**  
 * @brief Kernel to convert a single-precision matrix to half-precision.  
 *  
 * @param m     Number of rows in the matrix.  
 * @param n     Number of columns in the matrix.  
 * @param as    Pointer to the device-side single-precision matrix.  
 * @param ldas  Leading dimension of the single-precision matrix.  
 * @param ah    Pointer to the device-side half-precision matrix.  
 * @param ldah  Leading dimension of the half-precision matrix.  
 */  
__global__ void s2h(long int m, long int n, float *as, int ldas, __half *ah, int ldah);  

template <typename T>  
__global__ void setEye(int m, int n, T *a, int lda) {  
    long int i = threadIdx.x + blockDim.x * blockIdx.x;  
    long int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n) {  
        if (i == j)  
            a[i + j * lda] = static_cast<T>(1.0);  
        else  
            a[i + j * lda] = static_cast<T>(0.0);  
    }  
}  

// /**  
//  * @brief Kernel to set a single-precision matrix to the identity matrix.  
//  *  
//  * @param m    Number of rows in the matrix.  
//  * @param n    Number of columns in the matrix.  
//  * @param a    Pointer to the device-side single-precision matrix.  
//  * @param lda  Leading dimension of the matrix.  
//  */  
// __global__ void setSEye(long int m, long int n, float *a, int lda);  


template <typename T>  
void print_to_file(T *d_arr, int m, int n, const char *file_name);  

// template void Utils::print_to_file<float>(float*, int, int, const char*);  
// template void Utils::print_to_file<double>(double*, int, int, const char*);  


// 计算 I - A，并将结果存储到 w  
__global__ void minusEye(long int m, long int n, float *a, long int lda, float *w, long ldw);  

// 将单精度浮点数转换为半精度  
__global__ void s2h_(long int m, long int n, float *as, long int ldas, __half *ah, long int ldah);  

// 将双精度浮点数转换为单精度  
__global__ void d2s_(long int m, long int n, double *ad, long int ldas, float *as, long int ldah);  

// 将单精度浮点数转换为双精度  
__global__ void s2d_(long int m, long int n, float *as, long int ldas, double *ad, long int ldah);  

// 设备端数据复制  
// __global__ void deviceCopy(long m, long n, double *dB, long ldb, double *dA, long lda);  

// 将设备端矩阵设置为零  
template <typename T> 
__global__ void setZero(long m, long n, T *I, long ldi);  

// // 复制数据并清零源矩阵（双精度）  
// template <typename T> 
// __global__ void copyAndClear(long int m, long int n, T *da, int lda, T *db, int ldb);  

// // 复制数据并清零源矩阵（半精度）  
// __global__ void halfCopyAndClear(long int m, long int n, __half *da, int lda, __half *db, int ldb);  

template <typename T>  
__global__ void setZero(long m, long n, T* I, long ldi)  
{  
    // Calculate the row and column indices  
    long i = blockIdx.x * blockDim.x + threadIdx.x;  
    long j = blockIdx.y * blockDim.y + threadIdx.y;  
  
    if (i < m && j < n)  
    {  
        I[i + j * ldi] = static_cast<T>(0.0);  
    }  
}  



// Templated kernel to copy data from da to db and set da to zero  
template <typename T>  
__global__ void copyAndClear(long m, long n, T* da, int lda, T* db, int ldb)  
{  
    // Calculate the row and column indices  
    long i = blockIdx.x * blockDim.x + threadIdx.x;  
    long j = blockIdx.y * blockDim.y + threadIdx.y;  
  
    if (i < m && j < n)  
    {  
        db[i + j * ldb] = da[i + j * lda];  
        da[i + j * lda] = static_cast<T>(0.0);  
    }  
}  

template <typename T>  
__global__ void deviceCopy(long m, long n, T *dB, long ldb, T *dA, long lda)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n)  
    {  
        dB[i + j * ldb] = dA[i + j * lda];  
    }  
}  



// 将半精度转换为单精度  
__global__ void h2s(long int m, long int n, __half *as, int ldas, double *ah, int ldah);  



// 矩阵乘法  
void sgemm(int m, int n, int k, float *dA, int lda, float *dB, int ldb, float *dC, int ldc, float alpha, float beta);  

// 读取矩阵数据并分配到设备内存  
double* readMatrix(const char* filename, int m, int n);  

// 复制上三角矩阵  
__global__ void copyUpperTriangular(float* dest, int ldd, const float* src, int lds, int dest_row_offset, int dest_col_offset);  

// 复制矩阵  
__global__ void copyMatrix(float* dest, int ldd, const float* src, int lds, int dest_row_offset, int dest_col_offset, int rows, int cols);  

// 检查 CUDA 错误（无参数）  
void checkCUDAError(const char *msg);  

// 检查 CUDA 错误（带 cudaError_t 参数）  
void checkCudaError(cudaError_t err, const char *msg);  

// 检查 cuSolver 错误  
void checkCusolverError(cusolverStatus_t status, const char *msg);  

// // 内联设备函数：warp 级别的所有减少求和  
// __inline__ __device__ float warpAllReduceSum(float val);  

// cuBLAS nrm2 实现  
void nrm2(cublasHandle_t hdl, int n, const double *x, double *result);  

// cuBLAS axpy 实现  
cublasStatus_t axpy(cublasHandle_t handle, int n, double *alpha, const double *x, int incx, double *y, int incy);  

// 初始化 CUDA 事件  
void mystartTimer();  

// 停止计时并返回经过的毫秒数  
float mystopTimer();  
__global__ void setEye(double *I, long int n);

__global__ void setEye(float *I, long int n);
} // namespace Utils  

#endif // UTILS_H  