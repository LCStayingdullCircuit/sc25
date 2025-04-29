// utils.cu  
#include "utils.h"  

namespace Utils {  

// check = true;

float Snorm(long int m, long int n, float *dA) {  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    float dn;  
    int incx = 1;  
    cublasSnrm2(handle, m * n, dA, incx, &dn);  
    cublasDestroy(handle);  
    return dn;  
}  

double dnorm(long int m, long int n, double *dA) {  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    double dn;  
    int incx = 1;  
    cublasDnrm2(handle, m * n, dA, incx, &dn);  
    cublasDestroy(handle);  
    return dn;  
}  

float SVnorm(int m, float *x) {  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    float dn;  
    cublasSnrm2(handle, m, x, 1, &dn);  
    cublasDestroy(handle);  
    return dn;  
}  

double dVnorm(long int m, double *x) {  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    double dn;  
    cublasDnrm2(handle, m, x, 1, &dn);  
    cublasDestroy(handle);  
    return dn;  
}  

__global__ void s2h(long int m, long int n, float *as, int ldas, __half *ah, int ldah) {  
    long int i = threadIdx.x + blockDim.x * blockIdx.x;  
    long int j = threadIdx.y + blockDim.y * blockIdx.y;  
    if (i < m && j < n) {  
        ah[i + j * ldah] = __float2half(as[i + j * ldas]);  
    }  
}  



// __global__ void setSEye(long int m, long int n, float *a, int lda) {  
//     long int i = threadIdx.x + blockDim.x * blockIdx.x;  
//     long int j = threadIdx.y + blockDim.y * blockIdx.y;  

//     if (i < m && j < n) {  
//         if (i == j)  
//             a[i + j * lda] = 1.0f;  
//         else  
//             a[i + j * lda] = 0.0f;  
//     }  
// }  

template <typename T>  
void print_to_file(T *d_arr, int m, int n, const char *file_name) {  
    // 创建一个主机端的向量来存储从设备复制的数据  
    std::vector<T> h_arr(m * n);  

    // 从设备内存复制到主机内存  
    cudaMemcpy(h_arr.data(), d_arr, m * n * sizeof(T), cudaMemcpyDeviceToHost);  

    // 打开文件进行写入  
    std::ofstream file(file_name);  
    if (file.is_open()) {  
        // 遍历矩阵并写入文件  
        for (int i = 0; i < m; ++i) {  
            for (int j = 0; j < n; ++j) {  
                file << h_arr[j * m + i] << " ";  
            }  
            file << "\n";  
        }  
        file.close();  
    } else {  
        std::cerr << "Unable to open file: " << file_name << std::endl;  
    }  
}  

// 显式实例化所需的模板类型  
template void Utils::print_to_file<float>(float*, int, int, const char*);  
template void Utils::print_to_file<double>(double*, int, int, const char*);  

// 计算 I - A，并将结果存储到 w  
__global__ void minusEye(long int m, long int n, float *a, long int lda, float *w, long ldw)  
{  
    long int i = threadIdx.x + blockDim.x * blockIdx.x;  
    long int j = threadIdx.y + blockDim.y * blockIdx.y;  

    // 条件：在矩阵合法索引范围内  
    if (i < m && j < n)  
    {  
        // 计算 I - A，对角线元素用 1 减，非对角线用 0 减  
        if (i == j)  
        {  
            a[i + j * lda] = 1.0f - a[i + j * lda];  
            w[i + j * ldw] = a[i + j * lda];  
        }  
        else  
        {  
            a[i + j * lda] = 0.0f - a[i + j * lda];  
            w[i + j * ldw] = a[i + j * lda];  
        }  
    }  
}  

// 将单精度浮点数转换为半精度  
__global__ void s2h_(long int m, long int n, float *as, long int ldas, __half *ah, long int ldah)  
{  
    long int i = threadIdx.x + blockDim.x * blockIdx.x;  
    long int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n)  
    {  
        ah[i + j * ldah] = __float2half(as[i + j * ldas]);  
    }  
}  

// 将双精度浮点数转换为单精度  
__global__ void d2s_(long int m, long int n, double *ad, long int ldas, float *as, long int ldah)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n)  
    {  
        as[i + j * ldah] = static_cast<float>(ad[i + j * ldas]);  
    }  
}  

// 将单精度浮点数转换为双精度  
__global__ void s2d_(long int m, long int n, float *as, long int ldas, double *ad, long int ldah)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n)  
    {  
        ad[i + j * ldah] = static_cast<double>(as[i + j * ldas]);  
    }  
}  




// template <typename T>  
// __global__ void setZero(long m, long n, T* I, long ldi)  
// {  
//     // Calculate the row and column indices  
//     long i = blockIdx.x * blockDim.x + threadIdx.x;  
//     long j = blockIdx.y * blockDim.y + threadIdx.y;  
  
//     if (i < m && j < n)  
//     {  
//         I[i + j * ldi] = static_cast<T>(0.0);  
//     }  
// }  

// template __global__ void Utils::setZero<float>(long m, long n, float* I, long ldi);
// template __global__ void Utils::setZero<double>(long m, long n, double* I, long ldi);

// // Templated kernel to copy data from da to db and set da to zero  
// template <typename T>  
// __global__ void copyAndClear(long m, long n, T* da, int lda, T* db, int ldb)  
// {  
//     // Calculate the row and column indices  
//     long i = blockIdx.x * blockDim.x + threadIdx.x;  
//     long j = blockIdx.y * blockDim.y + threadIdx.y;  
  
//     if (i < m && j < n)  
//     {  
//         db[i + j * ldb] = da[i + j * lda];  
//         da[i + j * lda] = static_cast<T>(0.0);  
//     }  
// }  

// template __global__ void Utils::copyAndClearM<float>(long m, long n, float* da, int lda, float* db, int ldb);
// template __global__ void Utils::copyAndClearM<double>(long m, long n, double* da, int lda, double* db, int ldb);

// 复制数据并清零源矩阵（半精度）  
__global__ void halfCopyAndClear(long int m, long int n, __half *da, int lda, __half *db, int ldb)  
{  
    int i = threadIdx.x + blockDim.x * blockIdx.x;  
    int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n)  
    {  
        db[i + j * ldb] = da[i + j * lda];  
        da[i + j * lda] = __float2half(0.0f);  
    }  
}  

// 将半精度转换为单精度  
__global__ void h2s(long int m, long int n, __half *as, int ldas, double *ah, int ldah)  
{  
    long int i = threadIdx.x + blockDim.x * blockIdx.x;  
    long int j = threadIdx.y + blockDim.y * blockIdx.y;  

    if (i < m && j < n)  
    {  
        ah[i + j * ldah] = __half2float(as[i + j * ldas]);  
    }  
}  


// 矩阵乘法  
void sgemm(int m, int n, int k, float *dA, int lda, float *dB, int ldb, float *dC, int ldc, float alpha, float beta)  
{  
    cublasHandle_t handle;  
    cublasCreate(&handle);  
    float sone = alpha;  
    float szero = beta;  
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                                       m, n, k,  
                                       &sone, dA, lda,  
                                       dB, ldb,  
                                       &szero, dC, ldc);  
    if (status != CUBLAS_STATUS_SUCCESS) {  
        fprintf(stderr, "CUBLAS Sgemm failed\n");  
    }  
    cublasDestroy(handle);  
}  

// 读取矩阵数据并分配到设备内存  
double* readMatrix(const char* filename, int m, int n) {    
    

    std::ifstream infile(filename);  
    if (!infile.is_open()) {  
        std::cerr << "无法打开文件 " << filename << std::endl;  
        exit(EXIT_FAILURE);  
    }  

    std::cout << "矩阵尺寸: " << m << " 行, " << n << " 列" << std::endl;  

    // 读取矩阵数据到主机内存  
    std::vector<double> h_matrix(m * n);  
    for (int i = 0; i < m * n; ++i) {  
        if (!(infile >> h_matrix[i])) {  
            std::cerr << "文件中数据不足，无法填充矩阵。" << std::endl;  
            infile.close();  
            exit(EXIT_FAILURE);  
        }  
    }  

    infile.close();  

    // 分配设备内存  
    double* devicePtr = nullptr;  
    cudaError_t err = cudaMalloc((void**)&devicePtr, m * n * sizeof(double));  
    if (err != cudaSuccess) {  
        std::cerr << "cudaMalloc 错误: " << cudaGetErrorString(err) << std::endl;  
        exit(EXIT_FAILURE);  
    }  

    // 将数据从主机内存复制到设备内存  
    err = cudaMemcpy(devicePtr, h_matrix.data(), m * n * sizeof(double), cudaMemcpyHostToDevice);  
    if (err != cudaSuccess) {  
        std::cerr << "cudaMemcpy 错误: " << cudaGetErrorString(err) << std::endl;  
        cudaFree(devicePtr);  
        exit(EXIT_FAILURE);  
    }  

    std::cout << "矩阵已成功分配到设备内存，并写入 " << filename << std::endl;  

    return devicePtr;  
}  

// 复制上三角矩阵  
__global__ void copyUpperTriangular(float* dest, int ldd, const float* src, int lds, int dest_row_offset, int dest_col_offset) {  
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  
    if (col >= row && row < lds && col < lds) {  
        dest[(dest_row_offset + row) + (dest_col_offset + col) * ldd] = src[row + col * lds];  
    }  
}  

// 复制矩阵  
__global__ void copyMatrix(float* dest, int ldd, const float* src, int lds, int dest_row_offset, int dest_col_offset, int rows, int cols) {  
    int row = blockIdx.y * blockDim.y + threadIdx.y; // 行索引  
    int col = blockIdx.x * blockDim.x + threadIdx.x; // 列索引  
    if (row < rows && col < cols) {  
        dest[(dest_row_offset + row) + (dest_col_offset + col) * ldd] = src[row + col * lds];  
    }  
}   
// 检查 CUDA 错误（无参数）  
void checkCUDAError(const char *msg)  
{  
    cudaError_t err = cudaGetLastError();  
    if (err != cudaSuccess)  
    {  
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;  
        exit(EXIT_FAILURE);  
    }  
}  

// 检查 CUDA 错误（带 cudaError_t 参数）  
void checkCudaError(cudaError_t err, const char *msg)  
{  
    if (err != cudaSuccess)  
    {  
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl;  
        exit(EXIT_FAILURE);  
    }  
}  

// 检查 cuSolver 错误  
void checkCusolverError(cusolverStatus_t status, const char *msg)  
{  
    if (status != CUSOLVER_STATUS_SUCCESS)  
    {  
        std::cerr << "CUSOLVER Error: " << msg << " (" << status << ")" << std::endl;  
        exit(EXIT_FAILURE);  
    }  
}  



// cuBLAS nrm2 实现  
void nrm2(cublasHandle_t hdl, int n, const double *x, double *result)  
{  
    cublasStatus_t status = cublasDnrm2(hdl, n, x, 1, result);  
    // 假设 CGLS_CUDA_CHECK_ERR 是一个宏，用于检查并处理错误  
    if (status != CUBLAS_STATUS_SUCCESS)  
    {  
        std::cerr << "cuBLAS nrm2 failed with status: " << status << std::endl;  
        exit(EXIT_FAILURE);  
    }  
}  

// cuBLAS axpy 实现  
cublasStatus_t axpy(cublasHandle_t handle, int n, double *alpha,  
                   const double *x, int incx, double *y, int incy)  
{  
    cublasStatus_t err = cublasDaxpy(handle, n, alpha, x, incx, y, incy);  
    // 假设 CGLS_CUDA_CHECK_ERR 是一个宏，用于检查并处理错误  
    if (err != CUBLAS_STATUS_SUCCESS)  
    {  
        std::cerr << "cuBLAS axpy failed with status: " << err << std::endl;  
        exit(EXIT_FAILURE);  
    }  
    return err;  
}  

cudaEvent_t start, stop;
void mystartTimer()
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

float mystopTimer()
{
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return milliseconds;
}
__global__ void setEye(double *I, long int n) {  
    // 获取当前线程的行和列索引  
    long int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引  
    long int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引  

    // 确保线程索引在矩阵维度范围内  
    if (row < n && col < n) {  
        if (row == col) {  
            I[row * n + col] = 1.0;  // 对角线元素设为1  
        } else {  
            I[row * n + col] = 0.0;  // 其他元素设为0  
        }  
    }  
}  

__global__ void setEye(float *I, long int n) {  
    // 获取当前线程的行和列索引  
    long int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引  
    long int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引  

    // 确保线程索引在矩阵维度范围内  
    if (row < n && col < n) {  
        if (row == col) {  
            I[row * n + col] = 1.0;  // 对角线元素设为1  
        } else {  
            I[row * n + col] = 0.0;  // 其他元素设为0  
        }  
    }  
}  


// #endif // UTILS_H  

} // namespace Utils  
