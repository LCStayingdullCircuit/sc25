
#pragma once

#include "TensorBLAS.h"
#include <iomanip>

// Macro Definitions
#define THREADS_PER_BLOCK 256

// External Variable Declarations
extern int m, n, nb;
extern int datatype, condition;
extern float ms;
extern bool check;
extern float kernel_ms;
extern float y_ms;
extern float dtrsm_ms;
extern float gemm_ms;

// Namespace for HOU_QR
namespace HOU_QR {

// Function Declarations

// CUDA Kernel to set a matrix to the identity matrix
__global__ void setEye(long int m, long int n, float *a, long int lda);

// Inline device function for warp-level reduction sum
__inline__ __device__ float warpAllReduceSum(float val);

// Templated CUDA Kernel for Householder QR decomposition
template <long int M, long int N>
__global__ void hou_kernel3(long int m, long int n, float *AA, long int lda, float *RR, long int ldr);

// Function to perform panel QR decomposition using the templated kernel
template <long int M, long int N>
void hou_caqr_panel(cublasHandle_t handle, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work);

// CUDA Kernel to compute I - A and store the result in a work matrix
__global__ void minusEye(long int m, long int n, float *a, long int lda, float *w, long ldw);

// Function to reconstruct Y from LU factorization
void reconstructY(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, long m, long n, float *dA, long lda, float *U, float *work, int *info);

// CUDA Kernel to extract the U matrix from LU factorization
__global__ void getU(int m, int n, float *a, int lda, float *u, int ldu);

// CUDA Kernel to extract the L matrix from LU factorization
__global__ void getL(int m, int n, float *a, int lda);

// CUDA Kernel to convert single-precision matrix to half-precision
__global__ void s2h_(long int m, long int n, float *as, long int ldas, __half *ah, long int ldah);

// CUDA Kernel to copy matrix data from device A to device B
__global__ void deviceCopy(long m, long n, float *dB, long ldb, float *dA, long lda);

// CUDA Kernel to set a matrix to zero
__global__ void setZero(long m, long n, float *I, long ldi);

// Additional Function Declarations

// Function to perform panel QR decomposition
void panelQR(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, long m, long n, float *A, long lda, float *W, long ldw, float *R, long ldr, float *work, int *info);

// CUDA Kernels for copying and clearing matrices
__global__ void copyAndClear(long int m, long int n, float *da, int lda, float *db, int ldb);
__global__ void halfCopyAndClear(long int m, long int n, __half* da, int lda, __half* db, int ldb);
__global__ void h2s(long int m, long int n, __half *as, long int ldas, float *ah, long int ldah);

// Function to generate Q matrix from QR decomposition
void dorgqr(int m, int n, float *W, int ldw, float *Y, int ldy, float *work);

// External C linkage for hou_qr2 function
extern "C" void hou_qr2(long m, long n, int nb, float* A, long lda, float* W, long int ldw, float* R, long ldr, float* work, long lwork);

} // namespace HOU_QR

