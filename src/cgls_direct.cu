#include "LATER.h"
#include "utils.h"
#include "matrix_generation.h"
#include "qr_decomposition.h"
#include <random> 

using namespace Utils;
#define threadsPerDim 16

typedef int INT;
long int m, n, nb, b;
int itera;
long int m1, n1;
int datatype, condition;
#ifndef CGLS_DISABLE_ERROR_CHECK
#define CGLS_CUDA_CHECK_ERR()                                                   \
    do                                                                          \
    {                                                                           \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            printf("%s:%d:%s\n ERROR_CUDA: %s\n", __FILE__, __LINE__, __func__, \
                   cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
#else
#define CGLS_CUDA_CHECK_ERR()
#endif


// 确认输入参数，不涉及数据类型
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
    datatype = atoi(argv[4]);
    condition = atoi(argv[5]);
    itera = atoi(argv[6]);
    return 0;
}

// Numeric limit epsilon for float, double, complex_float, and complex_double.
template <typename T>
double Epsilon();

template <>
inline double Epsilon<double>()
{
    return std::numeric_limits<double>::epsilon();
}

template <>
inline double Epsilon<cuDoubleComplex>()
{
    return std::numeric_limits<double>::epsilon();
}

template <>
inline double Epsilon<float>()
{
    return std::numeric_limits<float>::epsilon();
}

template <>
inline double Epsilon<cuFloatComplex>()
{
    return std::numeric_limits<float>::epsilon();
}

template <typename T>  
int Solve(cublasHandle_t handle,  
          const T* A,  
          const int m,  
          const int n,  
          const T* b,  
          T* x,  
          const double shift,  
          const double tol,  
          const int maxit,  
          bool quiet,  
          std::vector<double>& normsRatios)  
{  

    T *p = nullptr, *q = nullptr, *r = nullptr, *s = nullptr;  
    double gamma = 0.0, normp = 0.0, normq = 0.0, norms = 0.0, norms0 = 0.0;  
    double normx = 0.0, xmax = 0.0;  
    int k = 0, flag = -1, indefinite = 0;  
    const double kEps = Epsilon<T>();  
    const T kOne = static_cast<T>(1.0);  
    const T kZero = static_cast<T>(0.0);  
    const T kNeg1 = static_cast<T>(-1.0);  

    cudaMalloc(&p, n * sizeof(T));  
    cudaMalloc(&q, m * sizeof(T));  
    cudaMalloc(&r, m * sizeof(T));  
    cudaMalloc(&s, n * sizeof(T));  

    cudaMemcpy(r, b, m * sizeof(T), cudaMemcpyDeviceToDevice);  
    cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);  

    nrm2(handle, n, x, &normx);  
    cudaDeviceSynchronize();  

    if (normx > 0.0)  
    {  
        double alpha = -1.0, beta = 1.0;  
        cublasDgemv(handle, CUBLAS_OP_N,  
                    m, n,  
                    &alpha, (const double*)A, m,  
                    (const double*)x, 1,  
                    &beta, (double*)r, 1);  
    }  

    {  
        double alpha = 1.0, zero = 0.0;  
        cublasDgemv(handle, CUBLAS_OP_T,  
                    m, n,  
                    &alpha,  
                    (const double*)A, m,  
                    (const double*)r, 1,  
                    &zero,  
                    (double*)s, 1);  
    }  
    cudaDeviceSynchronize();  

    cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);  
    nrm2(handle, n, s, &norms);  
    nrm2(handle, n, x, &normx);  
    cudaDeviceSynchronize();  

    norms0 = norms;  
    gamma  = norms * norms;  
    xmax   = normx;  


    for (k = 0; k < maxit; ++k)  
    {  
        double alpha = 1.0, zero = 0.0;  
        cublasDgemv(handle, CUBLAS_OP_N,  
                    m, n,  
                    &alpha,  
                    (const double*)A, m,  
                    (const double*)p, 1,  
                    &zero,  
                    (double*)q, 1);  

        cudaDeviceSynchronize();  
        nrm2(handle, n, p, &normp);  
        nrm2(handle, m, q, &normq);  

        double delta = normq * normq + shift * normp * normp;  
        if (delta <= 0.0) indefinite = 1;  
        if (delta == 0.0) delta = kEps;  

        double alphaD = gamma / delta;  
        double negAlphaD = -(gamma / delta);  

        axpy(handle, n, &alphaD, p, 1, x, 1);  
        axpy(handle, m, &negAlphaD, q, 1, r, 1);  

        {  
            double alpha2 = 1.0, zero2 = 0.0;  
            cublasDgemv(handle, CUBLAS_OP_T,  
                        m, n,  
                        &alpha2,  
                        (const double*)A, m,  
                        (const double*)r, 1,  
                        &zero2,  
                        (double*)s, 1);  
        }  

        cudaDeviceSynchronize();  
        nrm2(handle, n, s, &norms);  

        double gammaOld = gamma;  
        gamma = norms * norms;  
        double betaD = gamma / gammaOld;  

        axpy(handle, n, &betaD, p, 1, s, 1);  
        cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);  

        nrm2(handle, n, x, &normx);  
        cudaDeviceSynchronize();  

        xmax = std::max(xmax, normx);  
        bool converged = (norms <= norms0 * tol) || (normx * tol >= 1.0);  
        normsRatios.push_back(norms / norms0);  
        if (converged)  
        {  
            // flag = 1;
            break;  
        }  

    }  

    if(k < maxit - 1) {  
        flag = 1;  
    }  
    else {  
        flag = -1;  
    }  
    printf("iteration times is %d\n", k);
    if(itera == 5) {
        std::string text = ".txt";  
        std::string resultString = "res" + std::to_string(datatype)+ "_" + std::to_string(condition)+ "_nonpre" + text;  
        std::ofstream outFile(resultString);  
        for (double res : normsRatios) {  
            outFile << res << std::endl;  
        }  
    }
    cudaFree(p);  
    cudaFree(q);  
    cudaFree(r);  
    cudaFree(s);  
    return flag;  
}  


int main(int argc, char *argv[])  
{  
    if (parseArguments(argc, argv) == -1)  
        return 0;  
    double *dtA;   
    cudaMalloc(&dtA, sizeof(double) * m * n);  
    generateMatrix(dtA, m, n, condition, datatype);  

    // float *A;           cudaMalloc(&A, sizeof(float) * m * n);  
    // float *R;           cudaMalloc(&R, sizeof(float) * n * n);  
    // float *work;        cudaMalloc(&work, sizeof(float) * m * n);  
    // __half *halfwork;   cudaMalloc(&halfwork, sizeof(__half) * m * n);  
    // double *dR;         cudaMalloc(&dR, sizeof(double) * n * n);  
    
    cusolverDnHandle_t cusolver_handle;  
    cublasHandle_t     cublas_handle;  
    cudaCtxt           ctxt;  
    cusolverDnCreate(&cusolver_handle);  
    cublasCreate(&cublas_handle);  
    cublasCreate(&ctxt.cublas_handle);  

    std::vector<double> b(m, 1.0);  
    std::vector<double> x(n, 0.0);  
    double *d_b, *d_x;  
    cudaMalloc(&d_b, sizeof(double) * m);  
    cudaMalloc(&d_x, sizeof(double) * n);  
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_x, x.data(), sizeof(double) * n, cudaMemcpyHostToDevice);  
    double shift = 0.0;  
    double tol = 1e-13;  
    bool quiet = false;  
    int Iternumber = 0;  
    std::vector<double> normsRatios;  
    mystartTimer();
    int flag = Solve<double>(cublas_handle, dtA, m, n, d_b, d_x, shift, tol, itera, quiet, normsRatios);  
    float time = mystopTimer();
    printf("direct iteration takes %.4f ms\n", time);
    // cudaFree(work);  
    // cudaFree(halfwork);   
    cudaFree(dtA);  
    // cudaFree(A);  
    // cudaFree(R);  
    cudaFree(d_b);  
    cudaFree(d_x);  
    // cudaFree(dR);  

    cublasDestroy(cublas_handle);  
    cusolverDnDestroy(cusolver_handle);  
    cublasDestroy(ctxt.cublas_handle);  

    return 0;  
}  