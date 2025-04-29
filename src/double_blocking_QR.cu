#include "LATER.h"
#include "utils.h"
#include "matrix_generation.h"
#include "qr_decomposition.h"
#include <random> 

using namespace Utils;
#define threadsPerDim 16

typedef int INT;
long int m, n, nb, b;
long int m1, n1;
int datatype, condition;

int parseArguments(int argc, char *argv[])
{
    if (argc < 7)
    {
        printf("Needs m, n and nb as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    b = atoi(argv[4]);
    datatype = atoi(argv[5]);
    condition = atoi(argv[6]);
    return 0;
}

void generateQ(  
    int m, int n, int nb,  
    double* d_W,       
    double* d_Y,        
    double* d_Q,              
    double* work,                      
    cublasHandle_t cublas_handle)  
{  
    int num_blocks = n / nb;  
    double alpha = 1.0;  
    double beta = 0.0;  
    double negalpha = -1.0;
    
    for (int k = num_blocks - 1; k >= 0; k--) {  
        int start_row = k * nb;  
        int current_m = m - start_row;
          
        double* W_k = d_W + start_row * m + start_row; 
        double* Y_k = d_Y + start_row * m + start_row; 
        double* Q_k = d_Q + start_row * m + start_row;

        cublasDgemm(cublas_handle,  
                    CUBLAS_OP_T, CUBLAS_OP_N,  
                    nb, current_m, current_m,  
                    &alpha, Y_k, m,  
                    Q_k, m,          
                    &beta, work, nb); 

        cublasDgemm(cublas_handle,  
                    CUBLAS_OP_N, CUBLAS_OP_N,  
                    current_m, current_m, nb,  
                    &negalpha, W_k, m,     
                    work, nb,              
                    &alpha, Q_k, m);       
    }  
}  

int main(int argc, char *argv[])  
{  
    if (parseArguments(argc, argv) == -1)  
        return 0;  

    int lda = m;  
    const int ldwork1 = m + 108 * nb / 2;  
    double *dtA;   cudaMalloc(&dtA, sizeof(double) * m * n);  
    generateMatrix(dtA, m, n, condition, datatype);  

    double *W;     cudaMalloc(&W, sizeof(double) * m * n);  
    double *d_Y;   CHECK_CUDA(cudaMalloc(&d_Y, sizeof(double) * m * n));  
    double *d_Q;   CHECK_CUDA(cudaMalloc(&d_Q, sizeof(double) * m * m));  
    double *R;     cudaMalloc(&R, sizeof(double) * n * n);  
    double *work1; CHECK_CUDA(cudaMalloc(&work1, ldwork1 * 1024 * sizeof(double)));  
    double *work2; CHECK_CUDA(cudaMalloc(&work2, m * n * sizeof(double)));  
    double *work3; CHECK_CUDA(cudaMalloc(&work3, m * nb * sizeof(double)));  
    CHECK_CUDA(cudaMemset(W,   0, m * n * sizeof(double)));  
    CHECK_CUDA(cudaMemset(d_Y, 0, m * n * sizeof(double)));  

    cusolverDnHandle_t cusolver_handle;  
    cublasHandle_t     cublas_handle;  
    cudaCtxt           ctxt;  

    cusolverDnCreate(&cusolver_handle);  
    cublasCreate(&cublas_handle);  
    cublasCreate(&ctxt.cublas_handle);  

    dim3 blockDim(threadsPerDim, threadsPerDim);  
    dim3 gridDim((m + threadsPerDim - 1) / threadsPerDim,  
                 (m + threadsPerDim - 1) / threadsPerDim);  

    mystartTimer();  

    IterativeQR(dtA, m, n, nb, b,  W, d_Y, lda, d_Q, R,  
                work1, work2, work3, ldwork1, m*n, cublas_handle);  

    setEye<<<gridDim, blockDim>>>(d_Q, m);  

    generateQ(m, n, nb, W, d_Y, d_Q, work2, ctxt.cublas_handle);  

    cudaDeviceSynchronize();  

    float milliseconds1 = mystopTimer();  

    printf("takes %f ms\n", milliseconds1);  
    printf("tflops = %.4f\n", 4.0 * n * n * (m - 1.0/3 * n) / milliseconds1 / 1e9);  

    cudaFree(dtA);  
    cudaFree(W);  
    cudaFree(d_Y);  
    cudaFree(d_Q);  
    cudaFree(R);  
    cudaFree(work1);  
    cudaFree(work2);  
    cudaFree(work3);  

    cublasDestroy(cublas_handle);  
    cusolverDnDestroy(cusolver_handle);  
    cublasDestroy(ctxt.cublas_handle);  

    return 0;  
}  