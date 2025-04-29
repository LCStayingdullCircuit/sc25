#include "LATER.h"
#include "utils.h"
#include "matrix_generation.h"
#include "qr_decomposition.h"

extern "C" void hou_qr2(long m, long n, int nb, float* A, long lda, float* W, long ldw, float* R, long ldr, float* work, long lwork);  

using namespace Utils;

// #define USE_HOU_QR



int m, n, nb;
int datatype, condition;


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



__inline__ __device__ float warpAllReduceSum(float val) {
    for (int mask = warpSize/2; mask > 0; mask /= 2)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

int main(int argc, char *argv[])
{
    if (parseArguments(argc, argv) == -1)
        return 0;

    double *dtA; 
    cudaMalloc(&dtA, sizeof(double) * m * n);
    // generateUniformMatrixDouble(dtA, m, n);
    generateMatrix(dtA, m, n, condition, datatype);
    // std::string str = "matrix_" + std::to_string(m) + std::to_string(condition) + "_" + std::to_string(datatype) + ".txt";
    // std::vector<char> vec(str.begin(), str.end());
    // vec.push_back('\0');  // 手动添加 null 终止符
    // char* cstr = vec.data();
    // print_to_file(dtA, m, n, cstr); 


    float *A;
    cudaMalloc(&A, sizeof(float) * m * n);
    float *work;
    cudaMalloc(&work, sizeof(float) * m * n);
    float *R;
    cudaMalloc(&R, sizeof(float) * n * n);
    float *W;
    cudaMalloc(&W, sizeof(float) * m * n);
    float *oriA;
    float *oriA_;
    double *dR;
    cudaMalloc(&dR, sizeof(double) * n * n);

    dim3 gridDim((m + 31) / 32, (n + 31) / 32);
    dim3 gridDim1((n + 31) / 32, (n + 31) / 32);
    dim3 blockDim(32, 32);
    // void d2s_(int m, int n, double *ad, int ldas, float *as, int ldah)
    d2s_<<<gridDim, blockDim>>>(m, n, dtA, m, A, m);
    
    cudaMalloc(&oriA, sizeof(float) * m * n);
    cudaMemcpy(oriA, A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
    cudaMalloc(&oriA_, sizeof(float) * m * n);
    cudaMemcpy(oriA_, A, sizeof(float) * m * n, cudaMemcpyDeviceToDevice);
    __half* halfwork;
    cudaMalloc(&halfwork, sizeof(__half) * m * n);

    int *info;
    cudaMalloc(&info, sizeof(int));

    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle);

    double done = 1.0;


    mystartTimer();

    // 调用 qr 函数  
#ifdef USE_HOU_QR 
    hou_qr2(m, n, nb, A, m, W, m, R, n, work, m*n);
#else  
    qr(ctxt, m, n, A, m, R, n, work, m * n, halfwork, m * n);  

#endif  

    float milliseconds = mystopTimer();

    std::cout << "Time taken by qr function: " << milliseconds << " ms" << std::endl;  

    //检测结果函数
    if (true)
    {
#ifdef USE_HOU_QR 
        // printf("测试输出R\n");
        // const char* matrixR = "matrixR.txt";
        // print_to_file(R, n, n, matrixR);
        
        printf("测试hou_qr\n"); 
        printf("W的范数: %f\n",Snorm(m, n, W));
        printf("R的范数: %f\n",Snorm(m, n, R));

        // dorgqr(m, n, W, m, A, m, work);
        checkResult2(m, n, oriA, m, W, m, R, n);
        // checkOrthogonal(cublas_handle, m, n, W, m, n);
        checkOtho(m, n, W, m);
        
#else  
        printf("测试mgs_qr\n");
        checkResult2(m, n, oriA, m, A, m, R, n);
        checkOtho(m, n, A, m);
#endif  
        
    }
    s2d_<<<gridDim1, blockDim>>>(n, n, R, n, dR, n);
    //检测R输出
    // const char* matrixR = "matrixR.txt";
    // print_to_file<float>(R, m, n, matrixR);
    // printf("测试输出R\n");


    //计算AR-1
    cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                m, n,
                &done, dR, n,
                dtA, m);

    // const char* matrixAR = "matrixAR-1_mgs.txt";
    // print_to_file(oriA_, m, n, matrixAR);
    //printf("测试输出AR\n");


    double* d_S;  
    double* d_U;  
    double* d_VT;  
    int lwork_svd = 0;  
    int* d_info;  

    // 分配设备内存  
    cudaMalloc(&d_info, sizeof(int));  
    cudaMalloc((void**)&d_S, std::min(m, n) * sizeof(double));  
    cudaMalloc((void**)&d_U, m * m * sizeof(double));  
    cudaMalloc((void**)&d_VT, n * n * sizeof(double));  

    // 查询工作空间大小  
    cusolverDnDgesvd_bufferSize(cusolver_handle, m, n, &lwork_svd);  

    // 分配工作空间  
    double* d_work_svd = nullptr;  
    cudaMalloc((void**)&d_work_svd, lwork_svd * sizeof(double));  

    // 设置SVD参数  
    signed char jobu = 'N';  // 不计算U矩阵  
    signed char jobvt = 'N'; // 不计算VT矩阵  

    // 执行SVD  
    cusolverStatus_t status = cusolverDnDgesvd(  
        cusolver_handle,  
        jobu,  
        jobvt,  
        m,  
        n,  
        dtA,    // 确保 oriA_ 是 double 类型的指针  
        m,  
        d_S,  
        d_U,  
        m,  
        d_VT,  
        n,  
        d_work_svd,  
        lwork_svd,  
        nullptr,  // 不需要使用全部的执行信息  
        d_info  
    );  

    // 输出状态  
    std::cout << status << std::endl;  

    // 复制信息到主机并检查  
    int svdinfo;  
    cudaMemcpy(&svdinfo, d_info, sizeof(int), cudaMemcpyDeviceToHost);  
    if (svdinfo != 0) {  
        std::cerr << "SVD 失败，info = " << svdinfo << std::endl;  
        // 释放设备内存（根据需要）  
        cudaFree(d_S);  
        cudaFree(d_U);  
        cudaFree(d_VT);  
        cudaFree(d_work_svd);  
        cudaFree(d_info);  
        return -1;  
    }  


    std::string str = "singular_" + std::to_string(condition) + "_" + std::to_string(datatype) + ".txt";
    std::vector<char> vec(str.begin(), str.end());
    vec.push_back('\0');  // 手动添加 null 终止符
    char* cstr = vec.data();
    printMatrixDeviceBlock(cstr, m, 1, d_S, m);
    print_to_file(d_S, m, 1, cstr); 


    cudaFree(d_S);  
    cudaFree(d_U);  
    cudaFree(d_VT);  
    cudaFree(d_work_svd); 
    cudaFree(d_info);
    
    // float h_S[m];
    // cudaMemcpy(h_S, d_S, m * sizeof(float), cudaMemcpyDeviceToHost);
    
    // std::ofstream ofs(filensvd);
    // if (!ofs.is_open()) {
    //     std::cerr << "Error opening file for writing" << std::endl;
    //     return -1;
    // }

    // for (int i = 0; i < m; ++i) {
    //     ofs << h_S[i] << std::endl;
    //     //fp
    // }
    // printf("时间为：%f\n",mytime);
    // printf("qr时间为：%f\n",qrtime);
    
}