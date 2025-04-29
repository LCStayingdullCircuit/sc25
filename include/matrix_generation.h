#ifndef MATRIX_GENERATION_H  
#define MATRIX_GENERATION_H  

#include "utils.h"  

// Additional CUDA headers not included in utils.h  
#include <curand.h>           // For CURAND generator functions  
#include <curand_kernel.h>    // For CURAND device functions (e.g., curandState)  

// C++ Standard Library headers not included in utils.h  
#include <cmath>              // For mathematical functions (pow, fabs)  
#include <iomanip>            // For std::setprecision  
#include <cstdio>             // For printf, FILE operations  
#include <cstdlib>            // For exit  
#include <ctime>              // For time()  


#define THREADS_PER_BLOCK 256


/**  
 * @brief 使用 CURAND 填充数组的内核函数。  
 *  
 * @param arr    指向要填充的数组的指针。  
 * @param size   数组的大小。  
 * @param value  用于生成随机数的基准值。  
 * @param large  标志，决定随机数的生成方式。  
 * @param seed   用于 CURAND 初始化的种子。  
 */  
__global__ void fillArrayCluster(double* arr, long int size, double value, bool large, unsigned long seed);  

/**  
 * @brief 生成几何空间的数组。  
 *  
 * @param arr    指向要填充的数组的指针。  
 * @param size   数组的大小。  
 * @param start  几何序列的起始值。  
 * @param end    几何序列的结束值。  
 */  
__global__ void geometricSpace(double* arr, long int size, double start, double end);  

/**  
 * @brief 生成算术空间的数组。  
 *  
 * @param arr    指向要填充的数组的指针。  
 * @param size   数组的大小。  
 * @param start  算术序列的起始值。  
 * @param end    算术序列的结束值。  
 */  
__global__ void arithmeticSpace(double* arr, long int size, double start, double end);  

/**  
 * @brief 设置数组中的随机值。  
 *  
 * @param arr        指向要填充的数组的指针。  
 * @param size       数组的大小。  
 * @param type       随机值的类型标志。  
 * @param rand_vals  指向预生成随机值数组的指针。  
 */  
__global__ void setRandomValues(double* arr, long int size, int type, double* rand_vals);  

/**  
 * @brief Checks if a double-precision matrix is orthogonal.  
 *  
 * @param handle CUBLAS handle.  
 * @param m      Number of rows in matrix A.  
 * @param n      Number of columns in matrix A.  
 * @param A      Pointer to the device-side double-precision matrix A.  
 * @param lda    Leading dimension of matrix A.  
 * @param lde    Leading dimension for computations.  
 */  
void checkDOrthogonal(cublasHandle_t handle, int m, int n, double *A, int lda, int lde);  

/**  
 * @brief Checks if a single-precision matrix is orthogonal.  
 *  
 * @param m   Number of rows in matrix Q.  
 * @param n   Number of columns in matrix Q.  
 * @param Q   Pointer to the device-side single-precision matrix Q.  
 * @param ldq Leading dimension of matrix Q.  
 */  
void checkOtho(long int m, long int n, float *Q, int ldq);  

/**  
 * @brief Generates an orthogonal matrix using QR decomposition.  
 *  
 * @param handle   CUBLAS handle.  
 * @param cusolverH CUSOLVER handle.  
 * @param d_random_matrix Pointer to the device-side double-precision random matrix.  
 * @param m        Number of rows.  
 * @param n        Number of columns.  
 * @param gen      CURAND generator.  
 * @param seed     Seed for random number generation.  
 */  
void generateOrthogonalMatrix(cublasHandle_t handle, cusolverDnHandle_t cusolverH, double *d_random_matrix, long int m, long int n, curandGenerator_t gen, unsigned long long seed);  

/**  
 * @brief Generates a matrix based on specified distribution and condition number.  
 *  
 * @param d_res_matrix Pointer to the device-side result matrix.  
 * @param m            Number of rows.  
 * @param n            Number of columns.  
 */  
void generateMatrix(double *d_res_matrix, int m, int n, long condition, int datatype);  


// 检查 Ax 结果（版本2）  
void checkAxResult2(int m, int n, double *A, int lda, double *b, double *d_x);  

// 检查 Ax 结果  
void checkAxResult(int m, int n, double *A, int lda, double *b, double *d_x, double *R);  

// 检查 QR 分解结果  
void checkResult2(int m, int n, float *A, int lda, float *Q, int ldq, float *R, int ldr);  

// 生成均匀分布的单精度矩阵  
void generateUniformMatrix(float *dA, int m, int n);  

// 生成均匀分布的双精度矩阵  
void generateUniformMatrixDouble(double *dA, int m, int n);  
void checkResult2(int m, int n, double *A, int lda, double *Q, int ldq, double *R, int ldr) ;
void checkOtho(long int m, long int n, double *Q, int ldq) ;



#endif // MATRIX_GENERATION_H