#include "utils.h"  
#include "qr_decomposition.h"  
#include "kernelReWY.h"
#include <math.h>  

using namespace Utils;
#define threadsPerDim 16

float ms = 0;
// check = true;
float kernel_ms = 0;
float y_ms = 0;
float dtrsm_ms = 0;
float gemm_ms = 0;

// CUDA 核函数实现  
__global__ void mgs_kernel2(long int m, long int n, float *AA, long int lda, float *RR, long int ldr)  
{  
    int mm = m - blockIdx.x * 256; // TB local number of rows  
    mm = (mm < 256) ? mm : 256;  

    const int mnmin = (mm < n) ? mm : n;  

    // load from global memory to shared memory  
    __shared__ float As[256], Rs[32 * 32];  

#define ldrs 32  
    float Ar[8]; // register files  

    // load block A into registers.  
#pragma unroll 4  
    for (int l = 0; l < 8; l++)  
    {  
        if (threadIdx.x + l * 32 < mm && threadIdx.y < mnmin)  
        {  
            Ar[l] = AA[blockIdx.x * 256 + threadIdx.x + l * 32 + threadIdx.y * lda];  
        }  
    }  

    __syncthreads();  

    for (int k = 0; k < mnmin; k++)  
    {  
        float nu = 0; // acc for norm  

        if (threadIdx.y == k)  
        {  
#pragma unroll 8  
            for (int l = 0; l < 8; l++)  
            {  
                nu += (threadIdx.x + l * 32 < mm) ? (Ar[l] * Ar[l]) : 0;  
            }  
            float normx = sqrt((warpAllReduceSum(nu)));  
            if (threadIdx.x == k)  
            {  
                Rs[k + k * ldrs] = normx;  
            }  
            float scale = 1.0f / normx;  
#pragma unroll 8  
            for (int l = 0; l < 8; l++)  
            {  
                if (threadIdx.x + l * 32 < mm)  
                {  
                    Ar[l] *= scale;  
                    As[threadIdx.x + l * 32] = Ar[l];  
                }  
            }  
        }  
        __syncthreads();  
        nu = 0;  
        if (threadIdx.y > k)  
        {  
#pragma unroll 8  
            for (int l = 0; l < 8; l++)  
            {  
                if (threadIdx.x + l * 32 < mm)  
                {  
                    nu += (As[threadIdx.x + l * 32] * Ar[l]);  
                }  
            }  
            float scale = (warpAllReduceSum(nu));  
#pragma unroll 8  
            for (int l = 0; l < 8; l++)  
            {  
                if (threadIdx.x + l * 32 < mm)  
                {  
                    Ar[l] -= As[threadIdx.x + l * 32] * scale;  
                }  
            }  
            if (threadIdx.x == k)  
                Rs[k + threadIdx.y * ldrs] = scale;  
        }  
        __syncthreads();  
    }  

#pragma unroll 8  
    for (int l = 0; l < 8; l++)  
    {  
        if (threadIdx.x + l * 32 < mm && threadIdx.y < mnmin)  
            AA[blockIdx.x * 256 + threadIdx.x + l * 32 + threadIdx.y * lda] = Ar[l];  
    }  
    if (threadIdx.x < mnmin && threadIdx.y < mnmin)  
        RR[blockIdx.x * 32 + threadIdx.x + threadIdx.y * ldr] =  
            (threadIdx.x <= threadIdx.y) ? Rs[threadIdx.x + threadIdx.y * ldrs] : 0;  
}  

__global__ void getU(int m, int n, float *a, int lda, float *u, int ldu)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  
    if (i < m && j < n)  
    {  
        if (i > j)  
            u[i + j * ldu] = 0;  
        else  
            u[i + j * ldu] = a[i + j * lda];  
    }  
}  

__global__ void getU(int m, int n, double *a, int lda, double *u, int ldu)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  
    if (i < m && j < n)  
    {  
        if (i > j)  
            u[i + j * ldu] = 0;  
        else  
            u[i + j * ldu] = a[i + j * lda];  
    }  
}  


__global__ void getL(int m, int n, float *a, int lda)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  
    if (i < m && j < n)  
    {  
        if (i < j)  
            a[i + j * lda] = 0;  
        else if (i == j)  
            a[i + j * lda] = 1;  
    }  
}  

// 主机函数实现  
void mgs_caqr_panel_256x32(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work)  
{  
    if (n != 32)  
    {  
        printf("[Error]: CAQR_32 does not support n!=32\n");  
        return;  
    }  

    if (m <= 256)  
    {  
        // 使用 dim3 来设置块维度  
        auto blockdim = dim3(32, 32);  
        mgs_kernel2<<<1, blockdim>>>(m, n, A, lda, R, ldr);  
    }  
    else  
    { // m > 256, 递归调用  
        float sone = 1.0f;  
        float szero = 0.0f;  

        if (m % 256 == 0)  
        {  
            int ldwork = m / 256 * 32;  
            int mm = m / 256 * 32;  
            auto blockdim = dim3(32, 32);  
            mgs_kernel2<<<m / 256, blockdim>>>(m, n, A, lda, work, ldwork);  

            mgs_caqr_panel_256x32(ctxt, mm, n, work, ldwork, R, ldr, work + ldwork * n);  

            cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                                      256, 32, 32,  
                                      &sone, A, lda, 256,  
                                      work, ldwork, 32,  
                                      &szero, A, lda, 256,  
                                      m / 256);  
        }  
        else  
        {  
            int nb = m / 256;  
            int r = m % 256;  
            int ldwork = m / 256 * 32 + 32;  
            int mm = m / 256 * 32 + 32;  
            auto blockdim = dim3(32, 32);  
            mgs_kernel2<<<m / 256 + 1, blockdim>>>(m, n, A, lda, work, ldwork);  

            mgs_caqr_panel_256x32(ctxt, mm, n, work, ldwork, R, ldr, work + ldwork * n);  
            cublasSgemmStridedBatched(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                                      256, 32, 32,  
                                      &sone, A, lda, 256,  
                                      work, ldwork, 32,  
                                      &szero, A, lda, 256,  
                                      nb);  

            cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                        r, 32, 32,  
                        &sone, A + nb * 256, lda,  
                        work + nb * 32, ldwork,  
                        &szero, A + nb * 256, lda);  
        }  
    }  
}  

void mgs_caqr_panel_256x128(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work)  
{  
    if (m < 256 || n != 128)  
    {  
        printf("CAQR_256x128: ERROR: m must be > 256, n must be 128. (m,n)=(%ld,%ld)\n", m, n);  
        return;  
    }  
    float sone = 1.0f;  
    float szero = 0.0f;  
    float snegone = -1.0f;  
    // QR 左半部分 64  
    mgs_caqr_panel_256x32(ctxt, m, 32, A, lda, R, ldr, work);  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,  
                32, 32, m,  
                &sone, A, lda,  
                &A[32 * lda], lda,  
                &szero, &R[32 * ldr], ldr);  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                m, 32, 32,  
                &snegone, A, lda,  
                &R[32 * ldr], ldr,  
                &sone, &A[32 * lda], lda);  
    mgs_caqr_panel_256x32(ctxt, m, 32, &A[32 * lda], lda, &R[32 + 32 * ldr], ldr, work);  
    // 更新后续 64  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,  
                64, 64, m,  
                &sone, A, lda,  
                &A[64 * lda], lda,  
                &szero, &R[64 * ldr], ldr);  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                m, 64, 64,  
                &snegone, A, lda,  
                &R[64 * ldr], ldr,  
                &sone, &A[64 * lda], lda);  
    // QR 右半部分 64  
    A = &A[64 * lda];  
    R = &R[64 * ldr + 64];  
    mgs_caqr_panel_256x32(ctxt, m, 32, A, lda, R, ldr, work);  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,  
                32, 32, m,  
                &sone, A, lda,  
                &A[32 * lda], lda,  
                &szero, &R[32 * ldr], ldr);  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                m, 32, 32,  
                &snegone, A, lda,  
                &R[32 * ldr], ldr,  
                &sone, &A[32 * lda], lda);  
    mgs_caqr_panel_256x32(ctxt, m, 32, &A[32 * lda], lda, &R[32 + 32 * ldr], ldr, work);  
}


void qr(cudaCtxt ctxt, long int m, long int n, float *A, long int lda, float *R, long int ldr, float *work, long int lwork, __half *hwork, long int lhwork)  
{  
    if (n <= NMIN)  
    {  
        // 假设 mgs_caqr_panel_256x128 已在其他地方定义  
        mgs_caqr_panel_256x128(ctxt, m, n, A, lda, R, ldr, work);  
        return;  
    }  

    // 左递归  
    qr(ctxt, m, n / 2, A, lda, R, ldr, work, lwork, hwork, lhwork);  
    float sone = 1.0f, szero = 0.0f;  
    float snegone = -1.0f;  
    __half hone = __float2half(1.0f);  
    __half hnegone = __float2half(-1.0f);  

    __half *Ah = hwork;  
    __half *Bh = &hwork[m * n / 2];  
    __half *temp_A;  
    cudaMalloc(&temp_A, m * n / 2 * sizeof(__half));  
    dim3 gridDim((m + 31) / 32, (n + 31) / 32);  
    dim3 blockDim(32, 32);  
    s2h<<<gridDim, blockDim>>>(m, n / 2, A, lda, Ah, m);  
    s2h<<<gridDim, blockDim>>>(m, n / 2, &A[n / 2 * lda], lda, Bh, m);  
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n / 2, n / 2, m,  
                &sone, Ah, CUDA_R_16F, lda, Bh, CUDA_R_16F, lda,  
                &szero, &R[n / 2 * ldr], CUDA_R_32F, ldr, CUDA_R_32F,  
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);  
    dim3 gridDim2((n + 31) / 32, (n + 31) / 32);  
    s2h<<<gridDim2, blockDim>>>(n / 2, n / 2, &R[n / 2 * ldr], ldr, Bh, n / 2);  
    cublasGemmEx(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n / 2, n / 2,  
                &snegone, Ah, CUDA_R_16F, m, Bh, CUDA_R_16F, n / 2,  
                &sone, &A[n / 2 * lda], CUDA_R_32F, lda, CUDA_R_32F,  
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);  

    qr(ctxt, m, n / 2, &A[n / 2 * lda], lda, &R[n / 2 + n / 2 * ldr], ldr, work, lwork, hwork, lhwork);  

    cudaFree(temp_A);  
    return;  
}  

void qr_float(cudaCtxt ctxt, long int m, long int n, float *A, long int lda,  
    float *R, long int ldr, float *work, long int lwork)  
{  
    if (n <= NMIN)  
    {  
    // 假设 mgs_caqr_panel_256x128 已在其他地方定义  
    mgs_caqr_panel_256x128(ctxt, m, n, A, lda, R, ldr, work);  
    return;  
    }  

    // 左递归  
    qr_float(ctxt, m, n / 2, A, lda, R, ldr, work, lwork);  

    float sone = 1.0f;  
    float szero = 0.0f;  
    float snegone = -1.0f;  

    // 计算 R[n / 2 * ldr] = A^T * (A + n/2*lda)  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,  
        n / 2, n / 2, m,  
        &sone, A, lda, A + (n / 2) * lda, lda,  
        &szero, R + (n / 2) * ldr, ldr);  

    // 更新 A + n/2*lda -= A * R[n / 2 * ldr]  
    cublasSgemm(ctxt.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,  
        m, n / 2, n / 2,  
        &snegone, A, lda, R + (n / 2) * ldr, ldr,  
        &sone, A + (n / 2) * lda, lda);  

    // 右递归  
    qr_float(ctxt, m, n / 2,  
    A + (n / 2) * lda, lda,  
    R + (n / 2) + (n / 2) * ldr, ldr,  
    work, lwork);  

    return;  
}  

void test_cusolverDnSorgqr2(long m, long n, float* d_A, long lda, float *R22)  
{  
    cusolverDnHandle_t cusolverH = NULL;  
    cublasHandle_t cublasH = NULL;  

    /* 设备内存 */  
    float *d_tau = nullptr;  
    int *d_info = nullptr;  
    float *d_work = nullptr;  

    int lwork_geqrf = 0;  
    int lwork_orgqr = 0;  
    int lwork = 0;  
    int info = 0;  

    // const float h_one = 1.0f;  
    // const float h_minus_one = -1.0f;  

    /* 步骤 1: 创建 cuSolver 和 cuBLAS 句柄 */  
    cusolverDnCreate(&cusolverH);  
    cublasCreate(&cublasH);  

    /* 步骤 2: 分配设备内存 */  
    cudaMalloc(reinterpret_cast<void **>(&d_tau), sizeof(float) * n);  
    cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));  

    /* 步骤 3: 查询 geqrf 和 orgqr 的工作空间 */  
    cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &lwork_geqrf);  
    cusolverDnSorgqr_bufferSize(cusolverH, m, n, n, d_A, lda, d_tau, &lwork_orgqr);  

    lwork = std::max(lwork_geqrf, lwork_orgqr);  
    cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork);  

    /* 步骤 4: 计算 QR 分解 */  
    cusolverDnSgeqrf(cusolverH, m, n, d_A, lda, d_tau, d_work, lwork, d_info);  

    /* 步骤 5: 检查 QR 是否成功 */  
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);  
    printf("after geqrf: info = %d\n", info);  
    if (info < 0) {  
        printf("%d-th parameter is wrong\n", -info);  
        exit(1);  
    }  
    dim3 gridDimR22((n + 31) / 32, (n + 31) / 32);  
    dim3 blockDim(32, 32);  
    copyUpperTriangular<<<gridDimR22, blockDim>>>(R22, n, d_A, lda, 0, 0);  

    /* 步骤 6: 计算 Q */  
    cusolverDnSorgqr(cusolverH, m, n, n, d_A, lda, d_tau, d_work, lwork, d_info);  

    /* 步骤 7: 检查 orgqr 是否成功 */  
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);  
    printf("after orgqr: info = %d\n", info);  
    if (info < 0) {  
        printf("%d-th parameter is wrong\n", -info);  
        exit(1);  
    }  

    /* 释放资源 */  
    cudaFree(d_tau);  
    cudaFree(d_info);  
    cudaFree(d_work);  

    cublasDestroy(cublasH);  
    cusolverDnDestroy(cusolverH);  
}  

void qr_anyrow2(cudaCtxt ctxt, long m, long n, float* A, long lda, float* R, long ldr, float* work, long lwork, __half* hwork, long lhwork) {  
    // 定义常数  
    const float sone = 1.0f;  
    const float szero = 0.0f;  
    const float snegone = -1.0f;  
    const dim3 blockDim(32, 32);  

    // 计算 n1 和 n2  
    int max_k = static_cast<int>(std::log2(std::floor(n / 128.0)));  
    int n1 = 128 * static_cast<int>(std::pow(2, max_k));  
    int n2 = n - n1;  
    printf("n1 = %d; n2 = %d\n", n1, n2);  

    // 如果 n2 为 0，直接调用 qr 分解  
    if (n2 == 0) {  
        qr(ctxt, m, n, A, lda, R, ldr, work, lwork, hwork, lhwork);  
        return;  
    }  

    // 分配和初始化矩阵 R11, R12, R22  
    float *R11 = nullptr, *R12 = nullptr, *R22 = nullptr;  
    cudaMalloc(&R11, sizeof(float) * n1 * n1);  
    cudaMalloc(&R12, sizeof(float) * n1 * n2);  
    cudaMalloc(&R22, sizeof(float) * n2 * n2);  
    cudaMemset(R11, 0, sizeof(float) * n1 * n1);  
    cudaMemset(R12, 0, sizeof(float) * n1 * n2);  
    cudaMemset(R22, 0, sizeof(float) * n2 * n2);  

    // 分配半精度矩阵 A1h, A2h  
    __half *A1h = nullptr, *A2h = nullptr;  
    cudaMalloc(&A1h, sizeof(__half) * m * n1);  
    cudaMalloc(&A2h, sizeof(__half) * m * n2);  

    // 分割矩阵 A 为 A1 和 A2  
    float* A1 = A;           // A1 指向 A 的前 n1 列  
    float* A2 = A + m * n1;  // A2 指向 A 的后 n2 列  

    // 对 A1 进行 QR 分解，结果存储在 A1 中，R11 存储 R 部分  
    qr(ctxt, m, n1, A1, lda, R11, n1, work, lwork, hwork, lhwork);  

    // 定义 CUDA 网格维度  
    dim3 gridDimA1((m + 31) / 32, (n1 + 31) / 32);  
    dim3 gridDimA2((m + 31) / 32, (n2 + 31) / 32);  

    // 将 A1 和 A2 从单精度转换为半精度，存储在 A1h 和 A2h 中  
    s2h_<<<gridDimA1, blockDim>>>(m, n1, A1, lda, A1h, m);  
    s2h_<<<gridDimA2, blockDim>>>(m, n2, A2, lda, A2h, m);  

    // 计算 R12 = A1h^T * A2h，并存储在 R12 中  
    // cublasStatus_t status;  
    cublasGemmEx(  
        ctxt.cublas_handle,  
        CUBLAS_OP_T, CUBLAS_OP_N,  
        n1, n2, m,  
        &sone,  
        A1h, CUDA_R_16F, m,  
        A2h, CUDA_R_16F, m,  
        &szero,  
        R12, CUDA_R_32F, n1,  
        CUBLAS_COMPUTE_32F,  
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  
    );  

    // 计算 A2 = A2 - A1 * R12  
    cublasGemmEx(  
        ctxt.cublas_handle,  
        CUBLAS_OP_N, CUBLAS_OP_N,  
        m, n2, n1,  
        &snegone,  
        A1, CUDA_R_32F, lda,  
        R12, CUDA_R_32F, n1,  
        &sone,  
        A2, CUDA_R_32F, lda,  
        CUDA_R_32F,  
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  
    );  

    test_cusolverDnSorgqr2(m, n2, A2, lda, R22);  

    // 拼接 R11, R12, R22 到 R  
    cudaMemset(R, 0, sizeof(float) * n * n);  

    // 复制 R11 到 R 的左上角  
    dim3 gridDimR11((n1 + 31) / 32, (n1 + 31) / 32);  
    copyUpperTriangular<<<gridDimR11, blockDim>>>(R, n, R11, n1, 0, 0);  

    // 复制 R12 到 R 的右上角  
    dim3 gridDimR12((n2 + 31) / 32, (n1 + 31) / 32);  
    copyMatrix<<<gridDimR12, blockDim>>>(R, n, R12, n1, 0, n1, n1, n2);  

    // 复制 R22 到 R 的右下角  
    dim3 gridDimR22_full((n2 + 31) / 32, (n2 + 31) / 32);  
    copyUpperTriangular<<<gridDimR22_full, blockDim>>>(R, n, R22, n2, n1, n1);  

    // 释放分配的设备内存  
    cudaFree(R11);  
    cudaFree(R12);  
    cudaFree(R22);  
    cudaFree(A1h);  
    cudaFree(A2h);  
}  


// 内联设备函数：warp 级别的所有减少求和  
__inline__ __device__ float warpAllReduceSum(float val)  
{  
    for (int mask = warpSize / 2; mask > 0; mask /= 2)  
        val += __shfl_xor_sync(0xffffffff, val, mask);  
    return val;  
}  

// double blocking
__global__ void copySubmatrixToFull(float* A, int lda, float* A2, int lda2, int m_A2, int n_A2, int posi_m, int posi_n) {  
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    if (col < n_A2) {  
        // 计算A2当前列的起始地址  
        float* A2_col = A2 + col * lda2;          
        // 计算A中目标位置当前列的起始地址  
        float* A_col = A + (posi_n + col) * lda + posi_m;              

        for (int row = 0; row < m_A2; ++row) {  
            A_col[row] = A2_col[row];  
        }  
    }  
}  

void TSQRSegment(float *d_A, float *d_W,   
    float *d_Y, float *d_R,  
    float *work1, float *work2,  
    int ldwork1, int lda,   
    int cols, int rows,        
    int b, cublasHandle_t cublas_handle  
   )  
{  
    if (cols == b) {  
        // --------------------------------  
        // 处理大小为 b 的子块  
        // --------------------------------  

        // 1. 做 TSQR  
        tsqr<float>(cublas_handle, rows, cols, d_A, lda, d_R, cols, work1, ldwork1);  

        // 2. ReconstructWY (求出 W、Y)  
        ReconstructWYKernel(d_A, lda, d_W, lda, d_Y, lda, work2, rows, cols);  
        cudaDeviceSynchronize();  

        // 3. 将 R 写回 A 的前 b×b 块 (上三角)  
        dim3 blockDim(threadsPerDim, threadsPerDim);  
        dim3 gridDim((cols + threadsPerDim - 1) / threadsPerDim, (cols + threadsPerDim - 1) / threadsPerDim);  
        getU<<<gridDim, blockDim>>>(cols, cols, d_R, cols, d_A, lda);  
        cudaDeviceSynchronize();  

    } else {  
        // --------------------------------  
        // 处理大小 i > b 的块：拆成前后两部分  
        // --------------------------------  
        int half_i = cols / 2;  
        float alpha = 1.0f;  
        float beta = 0.0f;  
        float minusAlpha = -1.0f;  

        // 1. 递归调用：处理前 half_i 列  
        TSQRSegment(d_A, d_W, d_Y, d_R,  
            work1, work2,  
            ldwork1, lda,  
            half_i, rows,  
            b, cublas_handle  
            );  

        // 2. 更新后面的 half_i 列  
        CHECK_CUBLAS(cublasSgemm(  
            cublas_handle,  
            CUBLAS_OP_T, // W'  
            CUBLAS_OP_N, // A  
            half_i,        
            half_i,        
            rows,          
            &alpha,  
            d_W, // W的起始地址  
            lda,         // lda  
            d_A + half_i * lda,     // A1的起始地址  
            lda,         // lda  
            &beta,  
            work1,     // temp的起始地址  
            half_i          
        ));  

        CHECK_CUBLAS(cublasSgemm(  
            cublas_handle,  
            CUBLAS_OP_N, // Y  
            CUBLAS_OP_N, // temp  
            rows,           
            half_i,        
            half_i,       
            &minusAlpha,  
            d_Y, // Y的起始地址  
            lda,         // lda  
            work1,       // temp的起始地址  
            half_i,         
            &alpha,  
            d_A + half_i * lda,   
            lda          // lda  
        ));  
        cudaDeviceSynchronize();  

        // 3. 递归调用：处理后 half_i 列  
        TSQRSegment(d_A + half_i + half_i * lda, d_W + half_i + half_i * lda, d_Y + half_i + half_i * lda, d_R,  
            work1, work2,  
            ldwork1, lda,  
            half_i, rows - half_i,  
            b, cublas_handle  
            );  

        // 4. 前、后两部分的 W、Y 拼接  
        CHECK_CUBLAS(cublasSgemm(  
            cublas_handle,  
            CUBLAS_OP_T, // Y1'  
            CUBLAS_OP_N, // W  
            half_i,        
            half_i,        
            rows,          
            &alpha,  
            d_Y,         // Y1的起始地址  
            lda,         // lda  
            d_W + half_i * lda, // W的起始地址  
            lda,         // lda  
            &beta,  
            work1,     // temp的起始地址  
            half_i            
        ));  
        
        CHECK_CUBLAS(cublasSgemm(  
            cublas_handle,  //  
            CUBLAS_OP_N, // W1  
            CUBLAS_OP_N, // temp  
            rows,           
            half_i,        
            half_i,       
            &minusAlpha,  
            d_W, // W1起始地址  
            lda,         // lda  
            work1,     // temp的起始地址  
            half_i,         
            &alpha,  
            d_W + half_i * lda,     // W的起始地址  
            lda          // lda  
        ));  
        cudaDeviceSynchronize();  
    }  
}  




void TSQRSegment(double *d_A, double *d_W,   
                 double *d_Y, double *d_R,  
                 double *work1, double *work2,  
                 int ldwork1, int lda,   
                 int cols, int rows,        
                 int b, cublasHandle_t cublas_handle  
) {  
    if (cols == b) {
        // --------------------------------
        // 处理大小为 b 的子块
        // -------------------------------- 
        // printf("1\n");
        // 1. 做 TSQR
        tsqr<double>(cublas_handle, rows, cols, d_A, lda, d_R, cols, work1, ldwork1);
        // CHECK_CUDA(cudaGetLastError());  
        // cudaDeviceSynchronize();  
        // 2. 重构 W 和 Y
        ReconstructWYKernel(d_A, lda, d_W, lda, d_Y, lda, work2, rows, cols);
        // cudaDeviceSynchronize();  
        // cudaDeviceSynchronize();

        // 3. 将 R 写回到 A 的前 b×b 块 (上三角)
        dim3 blockDim(threadsPerDim, threadsPerDim);  
        dim3 gridDim((cols + threadsPerDim - 1) / threadsPerDim, (cols + threadsPerDim - 1) / threadsPerDim);  
        getU<<<gridDim, blockDim>>>(cols, cols, d_R, cols, d_A, lda);  
        cudaDeviceSynchronize();

    } else { 
        // --------------------------------
        // 处理大小 i > b 的块：拆成前后两部分
        // -------------------------------- 
        int half_i = cols / 2;  
        double alpha = 1.0;  
        double beta = 0.0;  
        double minusAlpha = -1.0;  

        // 1. 递归调用：处理前 half_i 列
        TSQRSegment(d_A, d_W, d_Y, d_R,  
                    work1, work2,  
                    ldwork1, lda,  
                    half_i, rows,  
                    b, cublas_handle);

        // 2. 更新后面的 half_i 列
        CHECK_CUBLAS(cublasDgemm(  
            cublas_handle,  
            CUBLAS_OP_T, // W'  
            CUBLAS_OP_N, // A  
            half_i,        
            half_i,        
            rows,          
            &alpha,  
            d_W,           // W的起始地址  
            lda,           // lda  
            d_A + half_i * lda,     // A1的起始地址  
            lda,           // lda  
            &beta,  
            work1,         // temp的起始地址  
            half_i));  

        CHECK_CUBLAS(cublasDgemm(  
            cublas_handle,  
            CUBLAS_OP_N, // Y  
            CUBLAS_OP_N, // temp  
            rows,           
            half_i,        
            half_i,       
            &minusAlpha,  
            d_Y,           // Y的起始地址  
            lda,           // lda  
            work1,         // temp的起始地址  
            half_i,         
            &alpha,  
            d_A + half_i * lda,   
            lda));  
        cudaDeviceSynchronize();  

        // 3. 递归调用：处理后 half_i 列
        TSQRSegment(d_A + half_i + half_i * lda, d_W + half_i + half_i * lda, d_Y + half_i + half_i * lda, d_R,  
                    work1, work2,  
                    ldwork1, lda,  
                    half_i, rows - half_i,  
                    b, cublas_handle);

        // 4. 前、后两部分的 W、Y 拼接
        CHECK_CUBLAS(cublasDgemm(  
            cublas_handle,  
            CUBLAS_OP_T, // Y1'  
            CUBLAS_OP_N, // W  
            half_i,        
            half_i,        
            rows,          
            &alpha,  
            d_Y,           // Y1的起始地址  
            lda,           // lda  
            d_W + half_i * lda, // W的起始地址  
            lda,           // lda  
            &beta,  
            work1,         // temp的起始地址  
            half_i));  

        CHECK_CUBLAS(cublasDgemm(  
            cublas_handle,  
            CUBLAS_OP_N, // W1  
            CUBLAS_OP_N, // temp  
            rows,           
            half_i,        
            half_i,       
            &minusAlpha,  
            d_W,           // W1起始地址  
            lda,           // lda  
            work1,         // temp的起始地址  
            half_i,         
            &alpha,  
            d_W + half_i * lda,     // W的起始地址  
            lda));  
        cudaDeviceSynchronize();  
    }  
}  




void IterativeQR(float* d_A, int m, int n,   
                 int nb, int b,   
                 float* d_W, float* d_Y, int lda,     // dW，dY，dA的ld都是lda  
                 float* d_Q, float* d_R, float* work1,  // d_Q, d_R, work不需要lda，可以任意修改矩阵形状，只要矩阵大小不超过m*n  
                 float* work2, float* transW, int ldwork1, int ldwork2,  
                 cublasHandle_t cublas_handle) {  
    int currm = 0;  
    int currn = 0;  
    bool flag1 = true; // 用于判断 n 中的第一个 nb  
    int j;  
    float* d_W_current;   
    float* d_Y_current;   
    float* d_A_current;   

    // 将 while 循环改为 for 循环  
    for (j = 0; j * nb < n; j++) {  
        int nt = nb;  
        int i = b;  
        int prei = b;     // 保存上一个 i 值  
        bool flag = true; // 用于判断在 nb 内的第一个 b  
        float alpha = 1.0f;  
        float beta = 0.0f;  
        float minusAlpha = -1.0f;  

        while (i <= nt / 2) {  
            int rows = m - currm;  
            int cols = i;  

            // TSQR + ReconstructWY  
            TSQRSegment(d_A + currm + currn * lda, d_W + currm + currn * lda,  
                        d_Y + currm + currn * lda, d_R,  
                        work1, work2,   
                        ldwork1, lda,  
                        cols, rows,   
                        b, cublas_handle);  

            d_W_current = d_W + currm + currn * lda;   
            d_Y_current = d_Y + currm + currn * lda;   
            d_A_current = d_A + currm + currn * lda;   

            if (!flag) {  
                prei = i;  
                i *= 2;  
            }  

            // A = (I - W * Y')' * A  
            if (i != nt) {  
                CHECK_CUBLAS(cublasSgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // W'  
                    CUBLAS_OP_N, // A  
                    prei,        
                    nb - i,        
                    m - currm,          
                    &alpha,  
                    d_W_current, // W 的起始地址  
                    lda,         // lda  
                    d_A_current + prei * lda, // A1 的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,       // temp 的起始地址  
                    prei          
                ));  
                CHECK_CUBLAS(cublasSgemm(  
                    cublas_handle,  
                    CUBLAS_OP_N, // Y  
                    CUBLAS_OP_N, // temp  
                    m - currm,           
                    nb - i,        
                    prei,       
                    &minusAlpha,  
                    d_Y_current, // Y 的起始地址  
                    lda,         // lda  
                    work1,       // temp 的起始地址  
                    prei,         
                    &alpha,  
                    d_A_current + prei * lda,   
                    lda          // lda  
                ));  
                cudaDeviceSynchronize();  
            }  

            // 更新 W 矩阵  
            if (!flag) {  
                CHECK_CUBLAS(cublasSgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // Y1'  
                    CUBLAS_OP_N, // W  
                    currn - j * nb,        
                    prei,        
                    m - currm,          
                    &alpha,  
                    d_Y + currm + j * nb * lda, // Y1 的起始地址  
                    lda,         // lda  
                    d_W_current, // W 的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,       // temp 的起始地址  
                    currn - j * nb            
                ));  
                CHECK_CUBLAS(cublasSgemm(  
                    cublas_handle,  
                    CUBLAS_OP_N, // W1  
                    CUBLAS_OP_N, // temp  
                    m - j * nb,           
                    prei,        
                    currn - j * nb,       
                    &minusAlpha,  
                    d_W + j * nb + j * nb * lda, // W1 起始地址  
                    lda,         // lda  
                    work1,       // temp 的起始地址  
                    currn - j * nb,         
                    &alpha,  
                    d_W + j * nb + currn * lda,  // W 的起始地址  
                    lda          // lda  
                ));  
                cudaDeviceSynchronize();  
            } else {  
                flag = false;  
            }  

            currm += prei;         
            currn += prei;  
        }  

        if (currn != n) {  
            d_W_current = d_W + currm - nb + (currn - nb) * lda;   
            d_Y_current = d_Y + currm - nb + (currn - nb) * lda;    
            d_A_current = d_A + currm - nb + currn * lda;   

            CHECK_CUBLAS(cublasSgeam(  
                cublas_handle,  
                CUBLAS_OP_T, // 对矩阵 W 进行转置  
                CUBLAS_OP_N,   
                nb,          
                m - j * nb,  
                &alpha,      // alpha = 1  
                d_W_current, // 矩阵 W 的起始地址  
                m,           // 矩阵 W 的主维度  
                &beta,       // beta = 0  
                nullptr,     // 矩阵 B 不参与运算（为空）  
                m,           // 占位  
                transW,      // 转置后的矩阵 W 存储到 d_transW  
                nb           // 转置后结果矩阵的主维度  
            ));   

            CHECK_CUBLAS(cublasSgemm(  
                cublas_handle,  
                CUBLAS_OP_N,  
                CUBLAS_OP_N,  
                nb,        
                n - (j + 1) * nb,        
                m - j * nb,          
                &alpha,  
                transW,  
                nb,  
                d_A_current,  
                lda,  
                &beta,  
                work2,  
                nb            
            ));  

            CHECK_CUBLAS(cublasSgemm(  
                cublas_handle,  
                CUBLAS_OP_N,  
                CUBLAS_OP_N,  
                m - j * nb,           
                n - (j + 1) * nb,        
                nb,       
                &minusAlpha,  
                d_Y_current,  
                lda,  
                work2,  
                nb,         
                &alpha,  
                d_A_current,  
                lda          
            ));  
            cudaDeviceSynchronize();  
        }  

        if (!flag1) {  
            // 执行额外检测逻辑  
        } else {  
            flag1 = false;  
        }  
    }  
}  




void IterativeQR(double* d_A, int m, int n,   
                 int nb, int b,   
                 double* d_W, double* d_Y, int lda,     // dW，dY，dA的ld都是lda  
                 double* d_Q, double* d_R, double* work1,  // d_Q, d_R, work不需要lda，可以任意修改矩阵形状，只要矩阵大小不超过m*n  
                 double* work2, double* transW, int ldwork1, int ldwork2,  
                 cublasHandle_t cublas_handle) {  
    int currm = 0;  
    int currn = 0;  
    bool flag1 = true; // 用于判断 n 中的第一个 nb  
    int j;  
    double* d_W_current;   
    double* d_Y_current;   
    double* d_A_current;   

    // 将 while 循环改为 for 循环  
    for (j = 0; j * nb < n; j++) {  
        int nt = nb;  
        int i = b;  
        int prei = b;     // 保存上一个 i 值  
        bool flag = true; // 用于判断在 nb 内的第一个 b  
        double alpha = 1.0;  
        double beta  = 0.0;  
        double minusAlpha = -1.0;  

        while (i <= nt / 2) {  
            int rows = m - currm;  
            int cols = i;  

            // TSQR + ReconstructWY  
            TSQRSegment(d_A + currm + currn * lda, d_W + currm + currn * lda,  
                        d_Y + currm + currn * lda, d_R,  
                        work1, work2,   
                        ldwork1, lda,  
                        cols, rows,   
                        b, cublas_handle);  

            d_W_current = d_W + currm + currn * lda;   
            d_Y_current = d_Y + currm + currn * lda;   
            d_A_current = d_A + currm + currn * lda;   

            if (!flag) {  
                prei = i;  
                i *= 2;  
            }  

            // A = (I - W * Y')' * A  
            if (i != nt) {  
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // W'  
                    CUBLAS_OP_N, // A  
                    prei,        
                    nb - i,        
                    m - currm,          
                    &alpha,  
                    d_W_current, // W 的起始地址  
                    lda,         // lda  
                    d_A_current + prei * lda, // A1 的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,       // temp 的起始地址  
                    prei          
                ));  
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_N, // Y  
                    CUBLAS_OP_N, // temp  
                    m - currm,           
                    nb - i,        
                    prei,       
                    &minusAlpha,  
                    d_Y_current, // Y 的起始地址  
                    lda,         // lda  
                    work1,       // temp 的起始地址  
                    prei,         
                    &alpha,  
                    d_A_current + prei * lda,   
                    lda          // lda  
                ));  
                cudaDeviceSynchronize();  
            }  

            // 更新 W 矩阵  
            if (!flag) {  
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_T, // Y1'  
                    CUBLAS_OP_N, // W  
                    currn - j * nb,        
                    prei,        
                    m - currm,          
                    &alpha,  
                    d_Y + currm + j * nb * lda, // Y1 的起始地址  
                    lda,         // lda  
                    d_W_current, // W 的起始地址  
                    lda,         // lda  
                    &beta,  
                    work1,       // temp 的起始地址  
                    currn - j * nb            
                ));  
                CHECK_CUBLAS(cublasDgemm(  
                    cublas_handle,  
                    CUBLAS_OP_N, // W1  
                    CUBLAS_OP_N, // temp  
                    m - j * nb,           
                    prei,        
                    currn - j * nb,       
                    &minusAlpha,  
                    d_W + j * nb + j * nb * lda, // W1 起始地址  
                    lda,         // lda  
                    work1,       // temp 的起始地址  
                    currn - j * nb,         
                    &alpha,  
                    d_W + j * nb + currn * lda,  // W 的起始地址  
                    lda          // lda  
                ));  
                cudaDeviceSynchronize();  
            } else {  
                flag = false;  
            }  

            currm += prei;         
            currn += prei;  
        }  

        if (currn != n) {  
            d_W_current = d_W + currm - nb + (currn - nb) * lda;   
            d_Y_current = d_Y + currm - nb + (currn - nb) * lda;    
            d_A_current = d_A + currm - nb + currn * lda;   

            CHECK_CUBLAS(cublasDgeam(  
                cublas_handle,  
                CUBLAS_OP_T, // 对矩阵 W 进行转置  
                CUBLAS_OP_N,   
                nb,          
                m - j * nb,  
                &alpha,      // alpha = 1  
                d_W_current, // 矩阵 W 的起始地址  
                m,           // 矩阵 W 的主维度  
                &beta,       // beta = 0  
                nullptr,     // 矩阵 B 不参与运算（为空）  
                m,           // 占位  
                transW,      // 转置后的矩阵 W 存储到 d_transW  
                nb           // 转置后结果矩阵的主维度  
            ));   

            CHECK_CUBLAS(cublasDgemm(  
                cublas_handle,  
                CUBLAS_OP_N,  
                CUBLAS_OP_N,  
                nb,        
                n - (j + 1) * nb,        
                m - j * nb,          
                &alpha,  
                transW,  
                nb,  
                d_A_current,  
                lda,  
                &beta,  
                work2,  
                nb            
            ));  

            CHECK_CUBLAS(cublasDgemm(  
                cublas_handle,  
                CUBLAS_OP_N,  
                CUBLAS_OP_N,  
                m - j * nb,           
                n - (j + 1) * nb,        
                nb,       
                &minusAlpha,  
                d_Y_current,  
                lda,  
                work2,  
                nb,         
                &alpha,  
                d_A_current,  
                lda          
            ));  
            cudaDeviceSynchronize();  
        }  

        if (!flag1) {  
            // 这部分是为了生成Q
            // double* d_W2_current = d_W + j * nb + j * nb * lda;  
            // double* d_Y1_current = d_Y + j * nb;  
            // CHECK_CUBLAS(cublasDgemm(  
            //     cublas_handle,  
            //     CUBLAS_OP_T, // Y1'  
            //     CUBLAS_OP_N, // W1  
            //     j * nb,        
            //     nb,        
            //     m - j * nb,          
            //     &alpha,  // Ensure alpha is of double type  
            //     d_Y1_current, // Y1's starting address  
            //     lda,         // lda  
            //     d_W2_current, // W1's starting address  
            //     lda,         // lda  
            //     &beta,       // Ensure beta is of double type  
            //     work2,       // temp's starting address  
            //     j * nb            
            // ));  
            
            // CHECK_CUBLAS(cublasDgemm(  
            //     cublas_handle,  
            //     CUBLAS_OP_N, // W1   
            //     CUBLAS_OP_N, // temp  
            //     m,           
            //     nb,        
            //     j * nb,       
            //     &minusAlpha, // Ensure minusAlpha is of double type  
            //     d_W,         // Y's starting address  
            //     lda,          // lda  
            //     work2,       // temp's starting address  
            //     j * nb,         
            //     &alpha,      // Ensure alpha is of double type  
            //     d_W + j * nb * lda, // A_sub's starting address  
            //     lda           // lda  
            // ));  
            // cudaDeviceSynchronize();  
        } else {  
            flag1 = false;  
        }  
    }  
}  
