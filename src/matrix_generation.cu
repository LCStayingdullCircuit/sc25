#include "utils.h"  
#include "matrix_generation.h"  



using namespace Utils;

__global__ void fillArrayCluster(double* arr, long int size, double value, bool large, unsigned long seed) {  
    int idx = threadIdx.x + blockDim.x * blockIdx.x;  
    if (idx < size) {  
        // 初始化 CURAND 状态  
        curandState state;  
        curand_init(seed, idx, 0, &state);  

        double randVal;  
        if(!large) {  
            if (idx < 48) {  
                // 生成范围在 (value - 1, value) 内的随机数  
                randVal = curand_uniform(&state) * (1.1 - 1.0) + 1.0;  
                randVal += value - 1;  
            } else {  
                // 生成范围在 (1, 1.1) 内的随机数  
                randVal = curand_uniform(&state) * (1.1 - 1.0) + 1.0;  
            }  
            arr[idx] = randVal;  
        }  
        else {  
            if (idx > 48) {  
                // 生成范围在 (value - 1, value) 内的随机数  
                randVal = curand_uniform(&state) * (1.1 - 1.0) + 1.0;  
                randVal += value - 1;  
            } else {  
                // 生成范围在 (1, 1.1) 内的随机数  
                randVal = curand_uniform(&state) * (1.1 - 1.0) + 1.0;  
            }  
            arr[idx] = randVal;  
        }  
    }  
}  

__global__ void geometricSpace(double* arr, long int size, double start, double end) {  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  
    if (idx < size) {  
        double factor = pow(end / start, 1.0 / (size - 1));  
        arr[idx] = start * pow(factor, idx);  
    }  
}  

__global__ void arithmeticSpace(double* arr, long int size, double start, double end) {  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  
    if (idx < size) {  
        double step = (end - start) / static_cast<double>(size - 1);  
        arr[idx] = start + step * idx;  
    }  
}  

__global__ void setRandomValues(double* arr, long int size, int type, double* rand_vals) {  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;  
    if (idx < size) {  
        arr[idx] = fabs(rand_vals[idx]);  
    }  
}  

void checkDOrthogonal(cublasHandle_t handle, int m, int n, double *A, int lda, int lde) {  
    double *eye_matrix;  
    double *ATA;  
    double d_norm_res;  
    double dnegone = -1.0;  
    double done = 1.0;  
    double dzero = 0.0;  
    int incx = 1;  

    // Allocate memory for ATA  
    checkCudaError(cudaMalloc(&ATA, m * n * sizeof(double)), "cudaMalloc");  

    // Allocate memory for the identity matrix  
    checkCudaError(cudaMalloc(&eye_matrix, m * n * sizeof(double)), "cudaMalloc");  

    // Define block and grid sizes  
    dim3 threadsPerBlock(16, 16);  
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,  
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);  

    // Generate the identity matrix on the device  
    setEye<<<blocksPerGrid, threadsPerBlock>>>(m, n, eye_matrix, m);  
    cudaDeviceSynchronize();  

    // Compute the norm of the identity matrix  
    cublasDnrm2(handle, m * n, eye_matrix, incx, &d_norm_res);  
    std::cout << "Eye Matrix Norm Result: " << d_norm_res << std::endl;  

    // Copy the identity matrix back to host for verification (optional)  
    std::vector<double> h_eye_matrix(m * n);  
    cudaMemcpy(h_eye_matrix.data(), eye_matrix, m * n * sizeof(double), cudaMemcpyDeviceToHost);  

    // Compute ATA = A^T * A  
    cublasDgeam(handle,  
                CUBLAS_OP_T, CUBLAS_OP_N,  
                m, n,  
                &done,  
                A, m,  
                &dzero,  
                A, m,  
                ATA, n);  

    // Compute ATA * (-1) + I  
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, m, &dnegone, A, m, ATA, n, &done, eye_matrix, m);  

    // Compute the norm of (I - A^T * A)  
    cublasDnrm2(handle, m * n, eye_matrix, 1, &d_norm_res);  
    std::cout << std::scientific;  
    std::cout << std::setprecision(6);  
    std::cout << "Norm Result: " << d_norm_res / m << std::endl;  

    // Free allocated memory  
    cudaFree(ATA);  
    cudaFree(eye_matrix);  
}  

void checkOtho(long int m, long int n, float *Q, int ldq) {  
    float *I;  
    checkCudaError(cudaMalloc(&I, sizeof(float) * n * n), "cudaMalloc");  

    // Define grid and block sizes  
    dim3 grid((n + 31) / 32, (n + 31) / 32);  
    dim3 block(32, 32);  

    // Generate the identity matrix on the device  
    setEye<float><<<grid, block>>>(n, n, I, n);  
    cudaDeviceSynchronize();  

    // Compute ||I|| using Snorm  
    float normRes = Utils::Snorm(n, n, I);  
    // std::printf("||I|| = %.6e\n", normRes);  

    float fnegone = -1.0f;  
    float fone = 1.0f;  

    cublasHandle_t handle;  
    cublasCreate(&handle);  

    // Compute I - Q^T * Q  
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,  
                &fnegone, Q, CUDA_R_32F, ldq, Q, CUDA_R_32F, ldq,  
                &fone, I, CUDA_R_32F, n, CUDA_R_32F,  
                CUBLAS_GEMM_DEFAULT);  

    // Compute the norm of (I - Q^T * Q)  
    normRes = Utils::Snorm(n, n, I);  
    // std::printf("normRes = %.6e\n", normRes);  

    std::printf("||I - Q'*Q||/N = %.6e\n", normRes / n);  

    // Clean up  
    cudaFree(I);  
    cublasDestroy(handle);  
}  


void checkResult2(int m, int n, double *A, int lda, double *Q, int ldq, double *R, int ldr)  
{  
    double normA = dnorm(m, n, A);  
    double alpha = 1.0;  
    double beta = -1.0;  
    
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, n, &alpha, Q, ldq, R, ldr, &beta, A, lda);  

    // 计算 ||A - QR|| / ||A||  
    double normRes = dnorm(m, n, A);  
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n", normRes / normA);  

    // 销毁handle  
    cublasDestroy(handle);  
}  

void checkOtho(long int m, long int n, double *Q, int ldq) {  
    double *I;  
    cudaMalloc(&I, sizeof(double) * n * n);  

    dim3 grid((n + 31) / 32, (n + 31) / 32);  
    dim3 block(32, 32);  

    setEye<double><<<grid, block>>>(n, n, I, n);  
    cudaDeviceSynchronize();  

    double snegone = -1.0;  
    double sone = 1.0;  

    cublasHandle_t handle;  
    cublasCreate(&handle);  

    // 执行矩阵相乘  
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &snegone, Q, ldq, Q, ldq, &sone, I, n);  

    double normRes = dnorm(n, n, I);  
    printf("Orthogonal error: ||I - Q'*Q||/N = %.6e\n", normRes / n);  

    // 清理资源  
    cudaFree(I);  
    cublasDestroy(handle);  
}  


void generateOrthogonalMatrix(cublasHandle_t handle, cusolverDnHandle_t cusolverH, double *d_random_matrix, long int m, long int n, curandGenerator_t gen, unsigned long long seed) {  
    // double done = 1.0;  
    // double dzero = 0.0;  
    // double dnegone = -1.0;  

    // Initialize CURAND generator  
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);  
    curandSetPseudoRandomGeneratorSeed(gen, seed);  

    // Generate a random matrix with normal distribution  
    curandGenerateNormalDouble(gen, d_random_matrix, m * n, 0.0, 1.0);  
    curandDestroyGenerator(gen);  

    int *d_info;  
    checkCudaError(cudaMalloc(&d_info, sizeof(int)), "cudaMalloc");  

    int lwork = 0;  
    double *d_work = NULL;  

    // Query working space for QR decomposition  
    cusolverDnDgeqrf_bufferSize(cusolverH, m, n, d_random_matrix, m, &lwork);  
    checkCudaError(cudaMalloc(&d_work, lwork * sizeof(double)), "cudaMalloc");  

    double *tau;  
    checkCudaError(cudaMalloc(&tau, m * sizeof(double)), "cudaMalloc");  

    // Perform QR factorization  
    cusolverStatus_t status = cusolverDnDgeqrf(cusolverH, m, n, d_random_matrix, m, tau, d_work, lwork, d_info);  
    // std::cout << "cusolverDnDgeqrf status = " << status << std::endl;  

    // Query working space for generating orthogonal matrix  
    cusolverDnDorgqr_bufferSize(cusolverH, m, n, n, d_random_matrix, m, tau, &lwork);  
    checkCudaError(cudaMalloc(&d_work, lwork * sizeof(double)), "cudaMalloc");  

    // Generate the orthogonal matrix Q  
    status = cusolverDnDorgqr(cusolverH, m, n, n, d_random_matrix, m, tau, d_work, lwork, d_info);  
    // std::cout << "cusolverDnDorgqr status = " << status << std::endl;  

    // Check if QR decomposition was successful  
    int h_info = 0;  
    cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);  
    if (h_info != 0) {  
        std::cerr << "Generating orthogonal matrix failed with info = " << h_info << std::endl;  
        // Handle the error as needed  
    }  

    // Clean up  
    cudaFree(d_info);  
    cudaFree(d_work);  
    cudaFree(tau);  
}  

void generateMatrix(double *d_res_matrix, int m, int n, long condition, int datatype)  
{  
    int condition_number = condition;  
    int distribution_type = datatype;  
    double alpha = 1.0;  
    double beta = 0.0;  

    // 声明需要的变量  
    double *d_diag = nullptr;  
    // double *d_rand_vals = nullptr;  
    curandGenerator_t gen = nullptr;  
    std::ofstream myfile;  

    if (distribution_type >= 1 && distribution_type <= 4) {  
        // 对于datatype 1到4，保持原有逻辑  
        cudaMalloc(&d_diag, n * sizeof(double)); // 使用 n 来存储对角线元素  
    
        // cudaMalloc(&d_rand_vals, n * sizeof(double)); // 使用 n  
    
        switch (distribution_type)  
        {  
        case 1:  
            fillArrayCluster<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_diag, n, condition_number, true, time(NULL));  
            break;  
        case 2:  
            fillArrayCluster<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_diag, n, condition_number, false,  time(NULL));  
            break;  
        case 3:  
            geometricSpace<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_diag, n, 1.0, (double)condition_number);  
            break;  
        case 4:  
            arithmeticSpace<<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_diag, n, 1.0, (double)condition_number);  
            break;  
        }  
    
        // 输出d_diag  
        // const char* d_diagfile = "diag.txt";  
        // print_to_file<double>(d_diag, n, 1, d_diagfile);  
        // printf("测试输出diag\n");  
    
        // 初始化cublas和cusolver  
        cublasHandle_t handle;  
        cublasCreate(&handle);  
        cusolverDnHandle_t cusolverH = NULL;  
        cusolverDnCreate(&cusolverH);  
    
        double *d_random_matrix, *d_random_matrix1;  
        cudaMalloc(&d_random_matrix, m * n * sizeof(double));  // Q1 是 mxn  
        cudaMalloc(&d_random_matrix1, n * n * sizeof(double)); // Q2 是 nxn  
    
        // 生成两个正交矩阵，注意 Q1 尺寸是 m*m ，Q2 尺寸是 n*n  
        generateOrthogonalMatrix(handle, cusolverH, d_random_matrix, m, n, gen, 1234ULL);  
        generateOrthogonalMatrix(handle, cusolverH, d_random_matrix1, n, n, gen, 5678ULL);  
    
        double *work;  
        cudaMalloc(&work, m * n * sizeof(double));  
    
        dim3 threadsPerBlock(16, 16);  
        dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,  
                           (n + threadsPerBlock.y - 1) / threadsPerBlock.y);  
    
        // 乘法计算: Q1 * diag * Q2'  
        setEye<<<blocksPerGrid, threadsPerBlock>>>(m, n, work, m); // 设置 m x n 矩阵  
    
        // 设置对角矩阵：work = Q1 * diag  
        cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, m, n, d_random_matrix, m, d_diag, 1, work, m);  
    
        // 计算最终结果 d_res_matrix = work * Q2'  
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, n, &alpha, work, m, d_random_matrix1, n, &beta, d_res_matrix, m);  
    
        // 释放资源  
        cudaFree(work);  
        cudaFree(d_diag);  
        // cudaFree(d_rand_vals);  
        cudaFree(d_random_matrix);  
        cudaFree(d_random_matrix1);  
        cublasDestroy(handle);  
        cusolverDnDestroy(cusolverH);  
    } else if (distribution_type == 5 || distribution_type == 6) {  
        // 对于datatype 5和6，直接生成元素为正态或均匀分布的矩阵  
        // curandGenerator_t gen;  
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);  
        curandSetPseudoRandomGeneratorSeed(gen, 2345ULL);  

        if (distribution_type == 5) {  
            // 生成正态分布的矩阵  
            curandGenerateNormalDouble(gen, d_res_matrix, m * n, 0.0, 1.0);  
        } else { // distribution_type == 6  
            // 生成均匀分布的矩阵  
            curandGenerateUniformDouble(gen, d_res_matrix, m * n);  
        }  

        // 销毁生成器  
        curandDestroyGenerator(gen);  
    } else {  
        std::cerr << "Invalid distribution type" << std::endl;  
        return;  
    }  
}


void checkAxResult2(int m, int n, double *A, int lda, double *b, double *d_x)  
{  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    double negdone = -1.0;  
    double dzero = 0.0;  
    double done = 1.0;  

    double normb0 = dVnorm(m, b);  
    double *tempx;  
    cudaMalloc(&tempx, sizeof(double) * n);  

    // 计算 b = A * d_x - b  
    cublasDgemv(handle, CUBLAS_OP_N, m, n, &done, A, m, d_x, 1, &negdone, b, 1);  

    // 计算 d_x = A^T * b  
    cublasDgemv(handle, CUBLAS_OP_T, m, n, &done, A, m, b, 1, &dzero, d_x, 1);  

    // 计算 ||Ax - b|| 和 ||A||  
    double normsol = dVnorm(m, b);  
    double norma = dnorm(m, n, A);  

    printf("||Ax-b|| = %.32lf\n ||A||=%.32lf\n", normsol, norma);  

    // 计算 ||A^T*(Ax - b)|| / (||A^T|| * ||b||)  
    normsol = dVnorm(n, d_x);  
    printf("||A^T*(Ax-b)||/(||A^T||*||b||) = %.6e\n", normsol / (norma * normb0));  

    // 释放内存  
    cudaFree(tempx);  
    cublasDestroy(handle);  
}  

void checkAxResult(int m, int n, double *A, int lda, double *b, double *d_x, double *R)  
{  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    double negdone = -1.0;  
    double dzero = 0.0;  
    double done = 1.0;  

    // 求解 R * x = b  
    cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 1, &done, R, n, d_x, n);  

    double normx = dVnorm(n, d_x);  
    double normb0 = dVnorm(m, b);  
    double *tempx;  
    cudaMalloc(&tempx, sizeof(double) * n);  

    // tempx = A^T * b  
    cublasDgemv(handle, CUBLAS_OP_T, m, n, &done, A, m, b, 1, &dzero, tempx, 1);  
    double normb = dVnorm(n, tempx);  

    // 计算 b = A * d_x - b  
    cublasDgemv(handle, CUBLAS_OP_N, m, n, &done, A, m, d_x, 1, &negdone, b, 1);  

    // 计算 d_x = A^T * b  
    cublasDgemv(handle, CUBLAS_OP_T, m, n, &done, A, m, b, 1, &dzero, d_x, 1);  

    // 计算 ||Ax - b|| 和 ||A||  
    double normsol = dVnorm(m, b);  
    double norma = dnorm(m, n, A);  

    printf("||Ax-b|| = %.32lf\n ||A||=%.32lf\n", normsol, norma);  

    // 计算 ||A^T*(Ax - b)|| / (||A^T|| * ||b||)  
    normsol = dVnorm(n, d_x);  
    printf("||A^T*(Ax-b)||/(||A^T||*||b||) = %.6e\n", normsol / (norma * normb0));  

    // 打开文件并记录结果  
    FILE *outFile = fopen("cgls_backerror.txt", "a");  
    if (outFile)  
    {  
        // 注意：变量 'condition' 和 'datatype' 需要在函数参数或全局定义  
        extern int condition;  
        extern int datatype;  
        fprintf(outFile, "%d %d %d %d %.6e\n", m, n, condition, datatype, normsol / (norma * normb0));  
        fclose(outFile);  
    }  
    else  
    {  
        printf("Failed to open results file for writing.\n");  
    }  

    // 释放内存  
    cudaFree(tempx);  
    cublasDestroy(handle);  
}  

void checkResult2(int m, int n, float *A, int lda, float *Q, int ldq, float *R, int ldr)  
{  
    float normA = Snorm(m, n, A);  
    float alpha = 1.0f;  
    float beta = -1.0f;  

    mystartTimer();  
    sgemm(m, n, n, Q, ldq, R, ldr, A, lda, alpha, beta);  
    float ms = mystopTimer();  

    // 计算 ||A - QR|| / ||A||  
    float normRes = Snorm(m, n, A);  
    printf("Backward error: ||A-QR||/(||A||) = %.6e\n", normRes / normA);  
}  

void generateUniformMatrix(float *dA, int m, int n)  
{  
    curandGenerator_t gen;  
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);  
    int seed = 3000;  
    curandSetPseudoRandomGeneratorSeed(gen, seed);  
    curandGenerateUniform(gen, dA, static_cast<size_t>(m) * n);  
    curandDestroyGenerator(gen);  
}  

void generateUniformMatrixDouble(double *dA, int m, int n)  
{  
    curandGenerator_t gen;  
    // 创建CURAND伪随机数生成器  
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);  
    // 设置随机生成器的种子  
    int seed = 3000;  
    curandSetPseudoRandomGeneratorSeed(gen, seed);  
    // 生成均匀分布的双精度浮点数  
    curandGenerateUniformDouble(gen, dA, static_cast<size_t>(m) * n);  
    // 销毁随机数生成器  
    curandDestroyGenerator(gen);  
}  

