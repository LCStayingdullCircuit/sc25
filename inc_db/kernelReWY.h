#pragma once  

#include <cuda_runtime.h>  

#define kerWYMAX_N 32  
#define kerWY_TRSM_BLOCKDIM 64  

// 声明 GPU 核函数  
__global__ void kernelIminusQ_inplace(float* d_Q, int ldq, int m, int n);  
__global__ void kernelIminusQ_inplace(double* d_Q, int ldq, int m, int n);  

__global__ void kernelLUDecompSharedV1(float* d_Q, int ldq, float* d_Y, int ldy, float* d_U, int n);  
__global__ void kernelLUDecompSharedV1(double* d_Q, int ldq, double* d_Y, int ldy, double* d_U, int n);  

__global__ void kernelLUDecompSharedV2(float* d_Q, int ldq, float* d_Y, int ldy, float* d_U, int n);  
__global__ void kernelLUDecompSharedV2(double* d_Q, int ldq, double* d_Y, int ldy, double* d_U, int n);  

__global__ void kernelLUDecompGlobal(float* d_Q, int ldq, float* d_Y, int ldy, float* d_U, int n);  
__global__ void kernelLUDecompGlobal(double* d_Q, int ldq, double* d_Y, int ldy, double* d_U, int n); 

__global__ void kernelTrsmRightUpperWARP(const float* __restrict__ d_U, int ldu, float* d_Q, int ldq,  
                                         float* d_Y, int ldy, int m, int n, int offsetRow);  
__global__ void kernelTrsmRightUpperWARP(const double* __restrict__ d_U, int ldu, double* d_Q, int ldq,  
    double* d_Y, int ldy, int m, int n, int offsetRow); 

__global__ void kernelTrsmRightLowerT(const float* __restrict__ d_L, int ldL, float* d_W, int ldW,  
                                      float* d_Q, int ldq, int m, int n);  
__global__ void kernelTrsmRightLowerT(const double* __restrict__ d_L, int ldL, double* d_W, int ldW,  
double* d_Q, int ldq, int m, int n);  

// 声明外部接口函数  
void ReconstructWYKernel(float* d_Q, int ldq, float* d_W, int ldw, float* d_Y, int ldy, float* d_U, int m, int n);  
void ReconstructWYKernel(double* d_Q, int ldq, double* d_W, int ldw, double* d_Y, int ldy, double* d_U, int m, int n); 