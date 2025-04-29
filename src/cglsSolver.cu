#include "LATER.h"
#include "utils.h"
#include "matrix_generation.h"
#include "qr_decomposition.h"

using namespace Utils;

typedef int INT;
long int m, n, nb;
int datatype, condition;
// float ms = 0;
// check = true;
// float kernel_ms = 0;
// float y_ms = 0;
// float dtrsm_ms = 0;
// float gemm_ms = 0;


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

cusolverStatus_t solveSystemUsingCUSOLVERDD(  
    cusolverDnHandle_t handle, // cuSOLVER handle  
    int m,                     // Number of rows
    int n,                     // Number of columns of matrix A  
    int nrhs,                  // Number of right-hand sides  
    double *dA,                // Device pointer to matrix A  
    int ldda,                  // Leading dimension of A  
    double *dB,                // Device pointer to right-hand sides B  
    int lddb,                  // Leading dimension of B  
    double *dX,                // Device pointer to solution matrix X  
    int lddx,                   // Leading dimension of X  
    float millisecondsDH
) {  
    cusolverStatus_t status;  
    size_t lwork_bytes;  
    void *dWorkspace;  
    int *dinfo;  
    cusolver_int_t niters;  

    status = cusolverDnDDgels_bufferSize(  
            handle,   
            m, n, nrhs,  
            dA, ldda,  
            dB, lddb,  
            dX, lddx,  
            dWorkspace, &lwork_bytes
    );    
    if (status != CUSOLVER_STATUS_SUCCESS) {  
        printf("Failed to query buffer size\n");  
        //printf("%d\n",status);
        return status;  
    }  

    // Allocate workspace and error info  
    cudaMalloc(&dWorkspace, lwork_bytes);  
    cudaMalloc(&dinfo, sizeof(int));  

     // Create CUDA events  
    cudaEvent_t start, stop;  
    cudaEventCreate(&start);  
    cudaEventCreate(&stop);  

    // Record the start event  
    cudaEventRecord(start, 0);  
    // Solve the system using iterative refinement  
    status = cusolverDnDDgels(  
        handle,  
        m, n, nrhs,  
        dA, ldda,  
        dB, lddb,  
        dX, lddx,  
        dWorkspace, lwork_bytes,  
        &niters,
        dinfo  
    );  
    printf("DD的niters = %d\n",niters);

    // Record the stop event  
    cudaEventRecord(stop, 0);  

    // Wait for the stop event to complete  
    cudaEventSynchronize(stop);  


    float milliseconds = 0;  
    cudaEventElapsedTime(&milliseconds, start, stop);  
    milliseconds = milliseconds + millisecondsDH;
    
    std::cout << "total Execution time: " << milliseconds << " ms" << std::endl;  
    std::ofstream outFile("time_cusolver.csv", std::ios::app);  
    if (!outFile.is_open()) {  
        std::cerr << "Error opening file for appending." << std::endl;  
        return status;  
    }  
    outFile << datatype << ',' << m << ',' << n << ',' << condition << ',' << milliseconds  << ',' << 2 * n / 1e9 * n * (m-n*1.0/3) / milliseconds << ',' << 1 << '\n';
      // 写入 norms 数据  
    outFile.close();  

    if (status != CUSOLVER_STATUS_SUCCESS) {  
        printf("status = %d\n", status);
        printf("Failed to solve the system\n");  
    } else {  
        int dinfo_h;  
        cudaMemcpy(&dinfo_h, dinfo, sizeof(int), cudaMemcpyDeviceToHost);  
        if (dinfo_h != 0) {  
            printf("Solution failed, dinfo = %d\n", dinfo_h);  
        } else {  
            printf("Solution was successful\n");  
            //printf("niters = %d\n",niters);
        }  
    }  
    //status = cusolverDnIRSInfosGetNiters( gesv_irs_infos, &niters );
    
    // Free workspace and infos  
    cudaFree(dWorkspace);  
    cudaFree(dinfo);  
    return status;  
}  
cusolverStatus_t solveSystemUsingXgels(  
    cusolverDnHandle_t handle, // cuSOLVER handle  
    int m,                     // Number of rows
    int n,                     // Number of columns of matrix A  
    int nrhs,                  // Number of right-hand sides  
    double *dA,                // Device pointer to matrix A  
    int ldda,                  // Leading dimension of A  
    double *dB,                // Device pointer to right-hand sides B  
    int lddb,                  // Leading dimension of B  
    double *dX,                // Device pointer to solution matrix X  
    int lddx                   // Leading dimension of X  
) {  
    
    double *dX2;
    cudaMalloc(&dX2, n * sizeof(double)); 
    cudaMemcpy(dX2, dX, n, cudaMemcpyDeviceToDevice);
    double *dA2;
    cudaMalloc(&dA2, m * n * sizeof(double)); 
    cudaMemcpy(dA2, dA, n, cudaMemcpyDeviceToDevice);
    cusolverStatus_t status;  
    size_t lwork_bytes;  
    void *dWorkspace;  
    int *dinfo;  
    cusolver_int_t niters;  

    status = cusolverDnDHgels_bufferSize(  
            handle,   
            m, n, nrhs,  
            dA, ldda,  
            dB, lddb,  
            dX, lddx,  
            dWorkspace, &lwork_bytes
    );    
    if (status != CUSOLVER_STATUS_SUCCESS) {  
        printf("Failed to query buffer size\n");  
        //printf("%d\n",status);
        return status;  
    }  

    // Allocate workspace and error info  
    cudaMalloc(&dWorkspace, lwork_bytes);  
    cudaMalloc(&dinfo, sizeof(int));  

     // Create CUDA events  
    cudaEvent_t start, stop;  
    cudaEventCreate(&start);  
    cudaEventCreate(&stop);  

    // Record the start event  
    cudaEventRecord(start, 0);  
    // Solve the system using iterative refinement  
    status = cusolverDnDHgels(  
        handle,  
        m, n, nrhs,  
        dA, ldda,  
        dB, lddb,  
        dX, lddx,  
        dWorkspace, lwork_bytes,  
        &niters,
        dinfo  
    );  
    // Record the stop event  
    cudaEventRecord(stop, 0);  
    // Wait for the stop event to complete  
    cudaEventSynchronize(stop);  
    // Calculate the elapsed time  
    float milliseconds = 0;  
    cudaEventElapsedTime(&milliseconds, start, stop);  
    printf("DH的niters = %d\n",niters);
    if(niters == -50){
        std::cout << "DH's Execution time: " << milliseconds << " ms" << std::endl;  
        cudaFree(dWorkspace);  
        cudaFree(dinfo);  
        status = solveSystemUsingCUSOLVERDD(handle, m, n, nrhs, dA, ldda, dB, lddb, dX2, lddx, milliseconds);  
        // checkAxResult2(m, n, dA, m, dB, dX2);
        //cudaMemcpy(dX, dX2, n, cudaMemcpyDeviceToDevice);
        return status; //这行代码其实不起作用
    }
    else if(niters == -5){
        printf("overflow occurred during computation\n");
        int info;
        cudaMemcpy(&info, dinfo, sizeof(int), cudaMemcpyDeviceToHost);
        printf("dinfo = %d\n", info);
        return status;
    }
    else{
        //首先验证后向误差
        checkAxResult2(m, n, dA2, m, dB, dX);
        // Print the elapsed time  
        std::cout << "DH's Execution time: " << milliseconds << " ms" << std::endl; 
        std::cout << "TFlops: " << 2 * n * n * (m-n*1.0/3)/milliseconds/1e9 << " TFlops" << std::endl;
        // 数据写入文件
        std::ofstream outFile("time_cusolver.csv", std::ios::app);  
        if (!outFile.is_open()) {  
            std::cerr << "Error opening file for appending." << std::endl;  
            return status;  
        }  
        outFile << datatype << ',' << m << ',' << n << ',' << condition << ',' << milliseconds << ',' << 2 * n /1e9 * n * (m-n*1.0/3)/milliseconds << ',' << 0 << '\n';
        // 写入 norms 数据  
        outFile.close();  
        }
        

        if (status != CUSOLVER_STATUS_SUCCESS) {  
            printf("status = %d\n", status);
            printf("Failed to solve the system\n");  
        } else {  
            int dinfo_h;  
            cudaMemcpy(&dinfo_h, dinfo, sizeof(int), cudaMemcpyDeviceToHost);  
            if (dinfo_h != 0) {  
                printf("Solution failed, dinfo = %d\n", dinfo_h);  
            } else {  
                printf("Solution was successful\n");  
                printf("niters = %d\n",niters);
            }  
        }  
        //status = cusolverDnIRSInfosGetNiters( gesv_irs_infos, &niters );
        
        // Free workspace and infos  
        cudaFree(dWorkspace);  
        cudaFree(dinfo);  
        return status;  
}  

cusolverStatus_t solveSystemUsingIRSXgels(  
    cusolverDnHandle_t handle, // cuSOLVER handle  
    int m,                     // Number of rows  
    int n,                     // Number of columns of matrix A  
    int nrhs,                  // Number of right-hand sides  
    double *dA,                // Device pointer to matrix A  
    int ldda,                  // Leading dimension of A  
    double *dB,                // Device pointer to right-hand sides B  
    int lddb,                  // Leading dimension of B  
    double *dX,                // Device pointer to solution matrix X  
    int lddx                   // Leading dimension of X  
) {  
    double *dA2;  
    CHECK_CUDA(cudaMalloc(&dA2, m * n * sizeof(double)));   
    CHECK_CUDA(cudaMemcpy(dA2, dA, m * n * sizeof(double), cudaMemcpyDeviceToDevice));  
    
    cusolverStatus_t status;  
    size_t lwork_bytes = 0;  
    void *dWorkspace = nullptr;  
    int *dinfo = nullptr;  
    cusolverDnIRSParams_t params = nullptr;  
    cusolverDnIRSInfos_t infos = nullptr;  
    float milliseconds = 0;  
    int dinfo_h = 0;  
    cusolver_int_t niters = 0;  
    
    
    cusolver_int_t m1 = m;  
    cusolver_int_t n1 = n;  
    cusolver_int_t nrhs1 = nrhs;  

    // 创建并初始化参数结构体  
    CHECK_CUSOLVER(cusolverDnIRSParamsCreate(&params));  

    // 创建并初始化信息结构体  
    CHECK_CUSOLVER(cusolverDnIRSInfosCreate(&infos));  

    // 设置主要精度和最低精度  
    CHECK_CUSOLVER(cusolverDnIRSParamsSetSolverPrecisions(  
        params,  
        CUSOLVER_R_64F, // 主要精度：double  
        CUSOLVER_R_16F  // 最低精度：half  
    ));  

    // 设置迭代收敛容差 tol = 1e-13  
    CHECK_CUSOLVER(cusolverDnIRSParamsSetTol(params, 1e-15));  

    // 设置最大迭代次数为 100  
    CHECK_CUSOLVER(cusolverDnIRSParamsSetMaxIters(params, 50));  

    // 设置迭代求精(之前貌似是没有设置这个？)
    CHECK_CUSOLVER(cusolverDnIRSParamsSetRefinementSolver(params, CUSOLVER_IRS_REFINE_GMRES));

    //设置内部迭代最大次数
    // CHECK_CUSOLVER(cusolverDnIRSParamsSetMaxItersInner(params, 10));

    //关闭回退高精度计算
    // CHECK_CUSOLVER(cusolverDnIRSParamsDisableFallback(params));

    // 查询所需的工作空间大小  
    CHECK_CUSOLVER(cusolverDnIRSXgels_bufferSize(  
        handle,  
        params,  
        m1, n1, nrhs1,  
        &lwork_bytes  
    ));  

    // 分配工作空间和错误信息  
    CHECK_CUDA(cudaMalloc(&dWorkspace, lwork_bytes));  
    CHECK_CUDA(cudaMalloc(&dinfo, sizeof(int)));  

    // 创建 CUDA 事件用于计时  
    cudaEvent_t start, stop;  
    CHECK_CUDA(cudaEventCreate(&start));  
    CHECK_CUDA(cudaEventCreate(&stop));  

    // 记录开始事件  
    CHECK_CUDA(cudaEventRecord(start, 0));  

    // 调用 cusolverDnIRSXgels 求解线性系统  
    CHECK_CUSOLVER(cusolverDnIRSXgels(  
        handle,  
        params,  
        infos,  
        m, n, nrhs,  
        dA, ldda,  
        dB, lddb,  
        dX, lddx,  
        dWorkspace,  
        lwork_bytes,  
        &niters,  
        dinfo  
    ));  

    // 记录结束事件  
    CHECK_CUDA(cudaEventRecord(stop, 0));  
    CHECK_CUDA(cudaEventSynchronize(stop));  

    // 计算执行时间  
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));  

    // 检查求解是否成功  
    CHECK_CUDA(cudaMemcpy(&dinfo_h, dinfo, sizeof(int), cudaMemcpyDeviceToHost));  
    if (dinfo_h != 0) {  
        std::cerr << "Solution failed, dinfo = " << dinfo_h << std::endl;  
        // Clean up allocated resources before exiting  
        cudaFree(dA2);  
        cudaFree(dWorkspace);  
        cudaFree(dinfo);  
        cusolverDnIRSInfosDestroy(infos);  
        cusolverDnIRSParamsDestroy(params);  
        cudaEventDestroy(start);  
        cudaEventDestroy(stop);  
        return CUSOLVER_STATUS_INTERNAL_ERROR;  
    } else {  
        std::cout << "Solution was successful" << std::endl;  
        checkAxResult2(m, n, dA2, m, dB, dX);  
    }  

    // 获取迭代次数  
    CHECK_CUSOLVER(cusolverDnIRSInfosGetNiters(infos, &niters));  
    std::cout << "Iteration count (niters) = " << niters << std::endl;  

    // int outerniters = 0;
    // CHECK_CUSOLVER(cusolverDnIRSInfosGetOuterNiters(infos, &niters));  
    // std::cout << "Outer iteration count (outerniters) = " << outerniters << std::endl;  

    // 输出执行时间  
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;  

    // 释放资源  
    cudaFree(dA2);  
    cudaFree(dWorkspace);  
    cudaFree(dinfo);  
    cusolverDnIRSInfosDestroy(infos);  
    cusolverDnIRSParamsDestroy(params);  
    cudaEventDestroy(start);  
    cudaEventDestroy(stop);  

    return CUSOLVER_STATUS_SUCCESS;  
}  



int main(int argc, char *argv[])
{
    if (parseArguments(argc, argv) == -1)
        return 0;

    // const char* matrixFile = "matrix_4884bcsstk16.txt";
    // double *dtA = readMatrix(matrixFile, 4884, 4884);  
    double *dtA;
    cudaMalloc(&dtA, sizeof(double) * m * n);
    generateMatrix(dtA, m, n, condition, datatype);
    cusolverDnHandle_t cusolver_handle;
    cublasHandle_t cublas_handle;
    cusolverDnCreate(&cusolver_handle);
    cublasCreate(&cublas_handle);
    cudaCtxt ctxt;
    cublasCreate(&ctxt.cublas_handle );

    
    std::vector<double> b(m);
    for(int i = 0; i < m; i++)
    {
        b[i] = 1.0;
    }
    std::vector<double> x(n);
    for(int i = 0; i < n; i++)
    {
        x[i] = 0.0;
    }
    double* d_b;
    double* d_x;
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    int nrhs = 1;
    int ldda = m;  
    int lddb = m;  
    int lddx = n;  
    // warmup
    for (int i = 0; i < 2; i++) { // 运行10次预热  
        generateMatrix(dtA, m, n, condition, datatype);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
        cudaMemcpy(d_x, x.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        solveSystemUsingIRSXgels(cusolver_handle, m, n, nrhs, dtA, ldda, d_b, lddb, d_x, lddx);  
    } 

    // solveSystemUsingIRSXgels(cusolver_handle, m, n, nrhs, dtA, ldda, d_b, lddb, d_x, lddx);  


}