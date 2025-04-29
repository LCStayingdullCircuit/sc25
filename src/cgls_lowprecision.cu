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
    if (argc < 6)
    {
        printf("Needs m, n and nb as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    datatype = atoi(argv[4]);
    condition = atoi(argv[5]);
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
int cgSolve(cublasHandle_t handle, const T* A, const T* R, const INT m, const INT n,  
             const T* b, T* x, const double shift, double tol,  
             const int maxit, bool quiet, int& Iternumber)  
{  
    T *p = nullptr, *q = nullptr, *r = nullptr, *s = nullptr;  
    T *tp = nullptr, *tx = nullptr;  
    double gamma = 0.0, normp = 0.0, normq = 0.0;  
    double norms = 0.0, norms0 = 0.0, normx = 0.0, xmax = 0.0;  
    std::vector<double> reslist;  
    char fmt[] = "%5d %9.2e %12.5g\n";  
    int k = 0, flag = -1, indefinite = 0;  

    const T  kOne    = static_cast<T>(1.0);  
    const T  kNegOne = static_cast<T>(-1.0);  
    const T  kZero   = static_cast<T>(0.0);  
    const double kEps = Epsilon<T>();  

    cudaMalloc(&p,  n * sizeof(T));  
    cudaMalloc(&q,  m * sizeof(T));  
    cudaMalloc(&r,  m * sizeof(T));  
    cudaMalloc(&s,  n * sizeof(T));  
    cudaMalloc(&tp, n * sizeof(T));  
    cudaMalloc(&tx, n * sizeof(T));  

    cudaMemcpy(r, b, m * sizeof(T), cudaMemcpyDeviceToDevice);  
    cudaMemcpy(s, x, n * sizeof(T), cudaMemcpyDeviceToDevice);  

    nrm2(handle, n, x, &normx);  
    cudaDeviceSynchronize();  

    if (normx > 0.0)  
    {  
        cudaMemcpy(tx, x, n * sizeof(T), cudaMemcpyDeviceToDevice);  
        cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,  
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  
                    n, 1, (const double*)&kOne,  
                    (const double*)R, n,  
                    (double*)tx, n);  
        double negOne = -1.0, one = 1.0;  
        cublasDgemv(handle, CUBLAS_OP_N, m, n,  
                    &negOne, (const double*)A, m,  
                    (const double*)tx, 1,  
                    &one,   (double*)r,  1);  
    }  

    {  
        double one = 1.0, zero = 0.0;  
        cublasDgemv(handle, CUBLAS_OP_T, m, n,  
                    &one,  (const double*)A, m,  
                    (const double*)r,        1,  
                    &zero, (double*)s,       1);  

        cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,  
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,  
                    n, 1, &one,  
                    (const double*)R, n,  
                    (double*)s,      n);  
    }  

    cudaDeviceSynchronize();  

    cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);  

    cudaDeviceSynchronize();  
    nrm2(handle, n, s, &norms);  
    norms0 = norms;  
    gamma = norms0 * norms0;  

    nrm2(handle, n, x, &normx);  
    xmax = normx;  

    for (k = 0; k < maxit; ++k)  
    {  
        cudaMemcpy(tp, p, n * sizeof(T), cudaMemcpyDeviceToDevice);  
        cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,  
                    CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,  
                    n, 1, (const double*)&kOne,  
                    (const double*)R, n,  
                    (double*)tp,      n);  
        {  
            double alpha = 1.0, beta = 0.0;  
            cublasDgemv(handle, CUBLAS_OP_N, m, n,  
                        &alpha, (const double*)A,  m,  
                        (const double*)tp,         1,  
                        &beta,  (double*)q,        1);  
        }  

        cudaDeviceSynchronize();  
        nrm2(handle, n, p, &normp);  
        nrm2(handle, m, q, &normq);  

        double delta = normq*normq + shift*normp*normp;  
        if (delta <= 0.) indefinite = 1;  
        if (delta == 0.) delta = kEps;  

        double alpha = (double)(gamma / delta);  
        double negalpha = -(double)(gamma / delta);  

        axpy(handle, n, &alpha,    p, 1, x, 1);  
        axpy(handle, m, &negalpha, q, 1, r, 1);  

        {  
            double one = 1.0, zero = 0.0;  
            cublasDgemv(handle, CUBLAS_OP_T, m, n,  
                        &one,  (const double*)A, m,  
                        (const double*)r,       1,  
                        &zero, (double*)s,      1);  

            cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,  
                        CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,  
                        n, 1, &one,  
                        (const double*)R, n,  
                        (double*)s,       n);  
        }  

        cudaDeviceSynchronize();  
        nrm2(handle, n, s, &norms);  
        double gammaOld = gamma;  
        gamma = norms * norms;  

        double beta = gamma / gammaOld;  
        axpy(handle, n, &beta, p, 1, s, 1);  
        cudaMemcpy(p, s, n * sizeof(T), cudaMemcpyDeviceToDevice);  

        nrm2(handle, n, x, &normx);  
        xmax = std::max(xmax, normx);  

        bool converged = (norms <= norms0 * tol) || (normx * tol >= 1.);  
        reslist.push_back(norms / norms0);  

        if (converged)  
        {  
            Iternumber = k;  
            break;  
        }  
    }  
    if(k < maxit - 1) {  
        flag = 1;  
    }  
    else {  
        flag = -1;  
    }  
    printf("Iteration times is %d\n", k);
    // std::string text = ".txt";  
    // std::string resultString = std::to_string(datatype)+ "_" + std::to_string(condition)+ "_pre" + text;  
    // std::ofstream outFile(resultString);  
    // for (double res : reslist) {  
    //     outFile << res << std::endl;  
    // }  
    // outFile.close();  

    cudaFree(p);  
    cudaFree(q);  
    cudaFree(r);  
    cudaFree(s);  
    cudaFree(tp);  
    cudaFree(tx);  

    return flag;  
}  


int main(int argc, char *argv[])  
{  
    if (parseArguments(argc, argv) == -1)  
        return 0;  
    double *dtA;   
    cudaMalloc(&dtA, sizeof(double) * m * n);  
    generateMatrix(dtA, m, n, condition, datatype);  

    float *A;           cudaMalloc(&A, sizeof(float) * m * n);  
    float *oriA;        cudaMalloc(&oriA, sizeof(float) * m * n);
    float *R;           cudaMalloc(&R, sizeof(float) * n * n);  
    float *work;        cudaMalloc(&work, sizeof(float) * m * n);  
    __half *halfwork;   cudaMalloc(&halfwork, sizeof(__half) * m * n);  
    double *dR;         cudaMalloc(&dR, sizeof(double) * n * n);  
    
    cusolverDnHandle_t cusolver_handle;  
    cublasHandle_t     cublas_handle;  
    cudaCtxt           ctxt;  
    cusolverDnCreate(&cusolver_handle);  
    cublasCreate(&cublas_handle);  
    cublasCreate(&ctxt.cublas_handle);  

    dim3 gridDim((m + 31) / 32, (n + 31) / 32);  
    dim3 blockDim(32, 32);  
    d2s_<<<gridDim, blockDim>>>(m, n, dtA, m, A, m);  
    cudaMemcpy(oriA, A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);  
    // warm up
    qr_anyrow2(  
        ctxt, m, n,  
        A, m,  
        R, n,  
        work, m * n,  
        halfwork, m * n  
    );  
    d2s_<<<gridDim, blockDim>>>(m, n, dtA, m, A, m);  
    cudaMemset(R, 0, sizeof(float) * n * n);  
    cudaMemset(work, 0, sizeof(float) * m * n);  
    cudaMemset(halfwork, 0, sizeof(__half) * m * n);

    
    mystartTimer();
    qr_anyrow2(  
        ctxt, m, n,  
        A, m,  
        R, n,  
        work, m * n,  
        halfwork, m * n  
    );  
    float lowQRTime = mystopTimer();
    checkResult2(m, n, oriA, m, A, m, R, n); 
    checkOtho(m, n, A, m);            

    std::vector<double> b(m, 1.0);  
    std::vector<double> x(n, 0.0);  
    double *d_b, *d_x;  
    cudaMalloc(&d_b, sizeof(double) * m);  
    cudaMalloc(&d_x, sizeof(double) * n);  
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);  
    cudaMemcpy(d_x, x.data(), sizeof(double) * n, cudaMemcpyHostToDevice);  
    
    dim3 gridDimR((n + 31) / 32, (n + 31) / 32);  
    s2d_<<<gridDimR, blockDim>>>(n, n, R, n, dR, n);  
    cudaDeviceSynchronize();  

    double shift = 0.0;  
    double tol = 1e-13;  
    int solvermaxit = 50;  
    int cgsovlermaxit = 50;  
    bool quiet = false;  
    int Iternumber = 0;  
    std::vector<double> normsRatios;  

    mystartTimer();
    int flag = cgSolve<double>(cublas_handle, dtA, dR, m, n, d_b, d_x, shift, tol, cgsovlermaxit, quiet, Iternumber);  
    cudaDeviceSynchronize();  
    float lowIterationtime = mystopTimer();
    printf("low precision QR time = %.4f, low precision Iteration time = %.4f\n", lowQRTime, lowIterationtime);
    cudaFree(work);  
    cudaFree(halfwork);  
    cudaFree(dtA);  
    cudaFree(A);  
    cudaFree(R);  
    cudaFree(d_b);  
    cudaFree(d_x);  
    cudaFree(dR);  

    cublasDestroy(cublas_handle);  
    cusolverDnDestroy(cusolver_handle);  
    cublasDestroy(ctxt.cublas_handle);  

    return flag;  
}  