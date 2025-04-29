#include "kernelReWY.h"  
#include <iostream>  

// 定义 GPU 核函数实现  
__global__ void kernelIminusQ_inplace(float* d_Q, int ldq, int m, int n) {  
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    int col = blockIdx.y;  
    if (row < m && col < n) {  
        float val = d_Q[col * ldq + row];  
        float valI = (row == col) ? 1.0f : 0.0f;  
        d_Q[col * ldq + row] = valI - val;  
    }  
}  

__global__ void kernelLUDecompSharedV1(float* d_Q, int ldq, float* d_Y, int ldy, float* d_U, int n) {  
    __shared__ float tile[kerWYMAX_N][kerWYMAX_N];  
    int row = threadIdx.x;  
    int col = threadIdx.y;  
    if (row < n && col < n) {  
        tile[row][col] = d_Q[col * ldq + row];  
    }  
    __syncthreads();  

    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  
        if (row > k && row < n && col == k) {  
            tile[row][col] /= tile[k][k];  
        }  
        __syncthreads();  
        if (row > k && row < n && col > k && col < n) {  
            tile[row][col] -= tile[row][k] * tile[k][col];  
        }  
    }  
    __syncthreads();  

    if (row < n && col < n) {  
        float val = tile[row][col];  
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row] = 0.f;  
        } else if (row == col) {  
            d_Y[col * ldy + row] = 1.f;  
            d_U[col * n + row] = val;  
        } else {  
            d_Y[col * ldy + row] = 0.f;  
            d_U[col * n + row] = val;  
        }  
    }  
}  

__global__ void kernelLUDecompSharedV2(float* d_Q, int ldq, float* d_Y, int ldy, float* d_U, int n) {  
    __shared__ float tile[kerWYMAX_N][kerWYMAX_N];  
    int row = threadIdx.x;  
    int col = threadIdx.y;  

    if (row < n && col < n) {  
        tile[col][row] = d_Q[col * ldq + row];  
    }  
    __syncthreads();  

    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  
        if (row > k && row < n && col == k) {  
            tile[col][row] /= tile[k][k];  
        }  
        __syncthreads();  
        if (row > k && row < n && col > k && col < n) {  
            tile[col][row] -= tile[k][row] * tile[col][k];  
        }  
    }  
    __syncthreads();  

    if (row < n && col < n) {  
        float val = tile[col][row];  
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row] = 0.f;  
        } else if (row == col) {  
            d_Y[col * ldy + row] = 1.f;  
            d_U[col * n + row] = val;  
        } else {  
            d_Y[col * ldy + row] = 0.f;  
            d_U[col * n + row] = val;  
        }  
    }  
}  

__global__ void kernelLUDecompGlobal(float* d_Q, int ldq, float* d_Y, int ldy, float* d_U, int n) {  
    int row = threadIdx.x;  
    int col = threadIdx.y;  

    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  

        if (row > k && row < n && col == k) {  
            d_Q[k * ldq + row] /= d_Q[k * ldq + k];  
        }  
        __syncthreads();  

        if (row > k && row < n && col > k && col < n) {  
            d_Q[col * ldq + row] -= d_Q[k * ldq + row] * d_Q[col * ldq + k];  
        }  
        __syncthreads();  
    }  

    if (row < n && col < n) {  
        float val = d_Q[col * ldq + row];  
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row] = 0.f;  
        } else if (row == col) {  
            d_Y[col * ldy + row] = 1.f;  
            d_U[col * n + row] = val;  
        } else {  
            d_Y[col * ldy + row] = 0.f;  
            d_U[col * n + row] = val;  
        }  
    }  
}  

__global__ void kernelTrsmRightUpperWARP(const float* __restrict__ d_U, int ldu, float* d_Q, int ldq,  
                                         float* d_Y, int ldy, int m, int n, int offsetRow) {  
    int rowIdxInBlock = threadIdx.x;  
    int globalRow = blockIdx.x * blockDim.x + rowIdxInBlock;  
    if (globalRow >= (m - offsetRow)) return;  

    int actualRow = globalRow + offsetRow;  
    extern __shared__ float sU[];  
    int t = threadIdx.x;  
    int stride = (m - 32 - blockIdx.x * blockDim.x) > blockDim.x ? blockDim.x : (m - 32 - blockIdx.x * blockDim.x);  
    for (int i = t; i < n * n; i += stride) {  
        sU[i] = __ldg(&d_U[i]);  
    }  
    __syncthreads();  

    float local[kerWYMAX_N];  
    for (int c = 0; c < n; c++) {  
        local[c] = d_Q[c * ldq + actualRow];  
    }  

    for (int c = 0; c < n; c++) {  
        float diag = sU[c * ldu + c];  
        local[c] /= diag;  

        #pragma unroll 4  
        for (int k = c + 1; k < n; k++) {  
            local[k] -= local[c] * sU[k * ldu + c];  
        }  
    }  

    for (int c = 0; c < n; c++) {  
        d_Y[c * ldy + actualRow] = local[c];  
    }  
}  

__global__ void kernelTrsmRightLowerT(const float* __restrict__ d_L, int ldL, float* d_W, int ldW,  
                                      float* d_Q, int ldq, int m, int n) {  
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    if (row >= m) return;  

    extern __shared__ float sLT[];  
    int t = threadIdx.x;  
    int stride = (m - blockIdx.x * blockDim.x) > blockDim.x ? blockDim.x : m - blockIdx.x * blockDim.x;  
    for (int i = t; i < n * n; i += stride) {  
        sLT[i] = __ldg(&d_L[(i / 32) * ldL + (i % 32)]);  
    }  
    __syncthreads();  

    float local[kerWYMAX_N];  
    for (int c = 0; c < n; ++c) {  
        local[c] = d_Q[c * ldq + row];  
    }  

    for (int c = 0; c < n; c++) {  
        float diag = sLT[c * n + c];  
        local[c] /= diag;  

        #pragma unroll 4  
        for (int k = c + 1; k < n; k++) {  
            local[k] -= local[c] * sLT[c * n + k];  
        }  
    }  

    for (int c = 0; c < n; ++c) {  
        d_W[c * ldW + row] = local[c];  
    }  
}  

void ReconstructWYKernel(float* d_Q, int ldq, float* d_W, int ldw, float* d_Y, int ldy, float* d_U, int m, int n) {  
    if (n > kerWYMAX_N) {  
        std::cerr << "[Error] n > " << kerWYMAX_N << " not supported.\n";  
        return;  
    }  

    dim3 block(32, 1);  
    dim3 grid((m + block.x - 1) / block.x, n);  
    kernelIminusQ_inplace<<<grid, block>>>(d_Q, ldq, m, n);  

    dim3 blockLU(n, n);  
    dim3 gridLU(1);  
    kernelLUDecompSharedV2<<<gridLU, blockLU>>>(d_Q, ldq, d_Y, ldy, d_U, n);  

    int rowsA2 = m - n;  
    if (rowsA2 > 0) {  
        int blockSize = kerWY_TRSM_BLOCKDIM;  
        dim3 gridTRSM((rowsA2 + blockSize - 1) / blockSize, 1);  
        size_t shmBytes = n * n * sizeof(float);  
        kernelTrsmRightUpperWARP<<<gridTRSM, blockSize, shmBytes>>>(d_U, n, d_Q, ldq, d_Y, ldy, m, n, n);  
    }  

    int blockSize = kerWY_TRSM_BLOCKDIM;  
    dim3 gridW((m + blockSize - 1) / blockSize, 1);  
    size_t shmBytes = n * n * sizeof(float);  
    kernelTrsmRightLowerT<<<gridW, blockSize, shmBytes>>>(d_Y, ldy, d_W, ldw, d_Q, ldq, m, n);  
}  


// 定义 GPU 核函数实现   
__global__ void kernelIminusQ_inplace(double* d_Q, int ldq, int m, int n) {  
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    int col = blockIdx.y;  
    if (row < m && col < n) {  
        double val = d_Q[col * ldq + row];  
        double valI = (row == col) ? 1.0 : 0.0;  
        d_Q[col * ldq + row] = valI - val;  
    }  
}  

__global__ void kernelLUDecompSharedV1(double* d_Q, int ldq, double* d_Y, int ldy, double* d_U, int n) {  
    __shared__ double tile[kerWYMAX_N][kerWYMAX_N];  
    int row = threadIdx.x;  
    int col = threadIdx.y;  
    if (row < n && col < n) {  
        tile[row][col] = d_Q[col * ldq + row];  
    }  
    __syncthreads();  

    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  
        if (row > k && row < n && col == k) {  
            tile[row][col] /= tile[k][k];  
        }  
        __syncthreads();  
        if (row > k && row < n && col > k && col < n) {  
            tile[row][col] -= tile[row][k] * tile[k][col];  
        }  
    }  
    __syncthreads();  

    if (row < n && col < n) {  
        double val = tile[row][col];  
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row] = 0.0;  
        } else if (row == col) {  
            d_Y[col * ldy + row] = 1.0;  
            d_U[col * n + row] = val;  
        } else {  
            d_Y[col * ldy + row] = 0.0;  
            d_U[col * n + row] = val;  
        }  
    }  
}  

__global__ void kernelLUDecompSharedV2(double* d_Q, int ldq, double* d_Y, int ldy, double* d_U, int n) {  
    __shared__ double tile[kerWYMAX_N][kerWYMAX_N];  
    int row = threadIdx.x;  
    int col = threadIdx.y;  

    if (row < n && col < n) {  
        tile[col][row] = d_Q[col * ldq + row];  
    }  
    __syncthreads();  

    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  
        if (row > k && row < n && col == k) {  
            tile[col][row] /= tile[k][k];  
        }  
        __syncthreads();  
        if (row > k && row < n && col > k && col < n) {  
            tile[col][row] -= tile[k][row] * tile[col][k];  
        }  
    }  
    __syncthreads();  

    if (row < n && col < n) {  
        double val = tile[col][row];  
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row] = 0.0;  
        } else if (row == col) {  
            d_Y[col * ldy + row] = 1.0;  
            d_U[col * n + row] = val;  
        } else {  
            d_Y[col * ldy + row] = 0.0;  
            d_U[col * n + row] = val;  
        }  
    }  
}  

__global__ void kernelLUDecompGlobal(double* d_Q, int ldq, double* d_Y, int ldy, double* d_U, int n) {  
    int row = threadIdx.x;  
    int col = threadIdx.y;  

    for (int k = 0; k < n - 1; k++) {  
        __syncthreads();  

        if (row > k && row < n && col == k) {  
            d_Q[k * ldq + row] /= d_Q[k * ldq + k];  
        }  
        __syncthreads();  

        if (row > k && row < n && col > k && col < n) {  
            d_Q[col * ldq + row] -= d_Q[k * ldq + row] * d_Q[col * ldq + k];  
        }  
        __syncthreads();  
    }  

    if (row < n && col < n) {  
        double val = d_Q[col * ldq + row];  
        if (row > col) {  
            d_Y[col * ldy + row] = val;  
            d_U[col * n + row] = 0.0;  
        } else if (row == col) {  
            d_Y[col * ldy + row] = 1.0;  
            d_U[col * n + row] = val;  
        } else {  
            d_Y[col * ldy + row] = 0.0;  
            d_U[col * n + row] = val;  
        }  
    }  
}  

__global__ void kernelTrsmRightUpperWARP(const double* __restrict__ d_U, int ldu, double* d_Q, int ldq,  
                                         double* d_Y, int ldy, int m, int n, int offsetRow) {  
    int rowIdxInBlock = threadIdx.x;  
    int globalRow = blockIdx.x * blockDim.x + rowIdxInBlock;  
    if (globalRow >= (m - offsetRow)) return;  

    int actualRow = globalRow + offsetRow;  
    extern __shared__ double sU1[];  
    int t = threadIdx.x;  
    int stride = (m - 32 - blockIdx.x * blockDim.x) > blockDim.x ? blockDim.x : (m - 32 - blockIdx.x * blockDim.x);  
    for (int i = t; i < n * n; i += stride) {  
        sU1[i] = __ldg(&d_U[i]);  
    }  
    __syncthreads();  

    double local[kerWYMAX_N];  
    for (int c = 0; c < n; c++) {  
        local[c] = d_Q[c * ldq + actualRow];  
    }  

    for (int c = 0; c < n; c++) {  
        double diag = sU1[c * ldu + c];  
        local[c] /= diag;  

        #pragma unroll 4  
        for (int k = c + 1; k < n; k++) {  
            local[k] -= local[c] * sU1[k * ldu + c];  
        }  
    }  

    for (int c = 0; c < n; c++) {  
        d_Y[c * ldy + actualRow] = local[c];  
    }  
}  

__global__ void kernelTrsmRightLowerT(const double* __restrict__ d_L, int ldL, double* d_W, int ldW,  
                                      double* d_Q, int ldq, int m, int n) {  
    int row = blockIdx.x * blockDim.x + threadIdx.x;  
    if (row >= m) return;  

    extern __shared__ double sLT1[];  
    int t = threadIdx.x;  
    int stride = (m - blockIdx.x * blockDim.x) > blockDim.x ? blockDim.x : m - blockIdx.x * blockDim.x;  
    for (int i = t; i < n * n; i += stride) {  
        sLT1[i] = __ldg(&d_L[(i / 32) * ldL + (i % 32)]);  
    }  
    __syncthreads();  

    double local[kerWYMAX_N];  
    for (int c = 0; c < n; ++c) {  
        local[c] = d_Q[c * ldq + row];  
    }  

    for (int c = 0; c < n; c++) {  
        double diag = sLT1[c * n + c];  
        local[c] /= diag;  

        #pragma unroll 4  
        for (int k = c + 1; k < n; k++) {  
            local[k] -= local[c] * sLT1[c * n + k];  
        }  
    }  

    for (int c = 0; c < n; ++c) {  
        d_W[c * ldW + row] = local[c];  
    }  
}  

void ReconstructWYKernel(double* d_Q, int ldq, double* d_W, int ldw, double* d_Y, int ldy, double* d_U, int m, int n) {  
    if (n > kerWYMAX_N) {  
        std::cerr << "[Error] n > " << kerWYMAX_N << " not supported.\n";  
        return;  
    }  

    dim3 block(32, 1);  
    dim3 grid((m + block.x - 1) / block.x, n);  
    kernelIminusQ_inplace<<<grid, block>>>(d_Q, ldq, m, n);  

    dim3 blockLU(n, n);  
    dim3 gridLU(1);  
    kernelLUDecompSharedV2<<<gridLU, blockLU>>>(d_Q, ldq, d_Y, ldy, d_U, n);  

    int rowsA2 = m - n;  
    if (rowsA2 > 0) {  
        int blockSize = kerWY_TRSM_BLOCKDIM;  
        dim3 gridTRSM((rowsA2 + blockSize - 1) / blockSize, 1);  
        size_t shmBytes = n * n * sizeof(double);  
        kernelTrsmRightUpperWARP<<<gridTRSM, blockSize, shmBytes>>>(d_U, n, d_Q, ldq, d_Y, ldy, m, n, n);  
    }  

    int blockSize = kerWY_TRSM_BLOCKDIM;  
    dim3 gridW((m + blockSize - 1) / blockSize, 1);  
    size_t shmBytes = n * n * sizeof(double);  
    kernelTrsmRightLowerT<<<gridW, blockSize, shmBytes>>>(d_Y, ldy, d_W, ldw, d_Q, ldq, m, n);  
}  