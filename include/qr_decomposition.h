#pragma once
// Include the Utils header for utility functions and definitions  
#include "utils.h"  
#include "LATER.h"



// Include CUDA-specific headers if not already included in utils.h  
#include <cuda_runtime.h>  
#include <cuda_fp16.h>

#include "TallShinnyQR.h"
#include "kernelReWY.h"


#define NMIN 128

// 定义 CUDA 上下文结构体  
// typedef struct {  
//     cublasHandle_t cublas_handle;  
//     // 你可以在这里添加更多的上下文相关信息  
// } cudaCtxt;  

//一些时间变量


// CUDA 核函数声明  
__global__ void mgs_kernel2(long int m, long int n, float *AA, long int lda, float *RR, long int ldr);  
__global__ void getU(int m, int n, float *a, int lda, float *u, int ldu);  
__global__ void getU(int m, int n, double *a, int lda, double *u, int ldu);  
__global__ void getL(int m, int n, float *a, int lda);  

// 主机函数声明  
void mgs_caqr_panel_256x32(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work);  
void mgs_caqr_panel_256x128(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work);  

void qr(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work, long int lwork, __half *hwork, long int lhwork);  
void qr_float(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work, long int lwork);

void test_cusolverDnSorgqr2(long m, long n, float* d_A, long lda, float *R22);  

void qr_anyrow2(cudaCtxt ctxt, long m, long n, float* A, long lda, float* R, long ldr, float* work, long lwork, __half* hwork, long lhwork);  

// double blocking
__global__ void copySubmatrixToFull(float* A, int lda, float* A2, int lda2, int m_A2, int n_A2, int posi_m, int posi_n);
// float
void TSQRSegment(float *d_A, float *d_W, 
    float *d_Y, float *d_R,  
    float *work1, float *work2,  
    int ldwork1,int lda, 
    int cols, int rows,        
    int b, cublasHandle_t cublas_handle
   );

// double 
void TSQRSegment(double *d_A, double *d_W,   
    double *d_Y, double *d_R,  
    double *work1, double *work2,  
    int ldwork1, int lda,   
    int cols, int rows,        
    int b, cublasHandle_t cublas_handle  
);
// float
void IterativeQR(float* d_A, int m, int n, 
    int nb, int b, 
    float* d_W, float* d_Y, int lda,     // dW，dY，dA的ld都是lda
    float* d_Q, float* d_R, float* work1,  //d_Q, d_R, work不需要lda，可以任意修改矩阵形状，只要矩阵大小不超过m*n
    float* work2, float* transW, int ldwork1, int ldwork2,
    cublasHandle_t cublas_handle);
// double
void IterativeQR(double* d_A, int m, int n,   
    int nb, int b,   
    double* d_W, double* d_Y, int lda,     // dW，dY，dA的ld都是lda  
    double* d_Q, double* d_R, double* work1,  // d_Q, d_R, work不需要lda，可以任意修改矩阵形状，只要矩阵大小不超过m*n  
    double* work2, double* transW, int ldwork1, int ldwork2,  
    cublasHandle_t cublas_handle);


__inline__ __device__ float warpAllReduceSum(float val);
