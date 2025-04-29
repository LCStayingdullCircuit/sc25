CXX = nvcc  


CFLAGS = -O3 -arch sm_90 -I $(CUDA_PATH)/include -I /yourpath/include \
         -I /yourpath/Tensor-BLAS/include \
         -I /yourpath/Tensor-BLAS/cuMpSGEMM/include  \
		 -I /yourpath/inc_db

LFLAGS = -L $(CUDA_PATH)/lib64 -lcusolver -lcublas -lcurand -lcudart -lcuda \
         -L /yourpath/Tensor-BLAS/build -lTensorBLAS \
         -L /yourpath/Tensor-BLAS/cuMpSGEMM/build -lcumpsgemm  

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin


SRCS = $(wildcard $(SRC_DIR)/*.cu)  

SOURCES = $(notdir $(SRCS))  

OBJECTS = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(SOURCES))  
  
BINS = halfAR_inv cglsSolver cgls_Iteration double_blocking_QR cgls_highprecision cgls_direct cgls_lowprecision
 
all: $(BINS)  


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu  
	@mkdir -p $(OBJ_DIR)  
	$(CXX) $(CFLAGS) -c $< -o $@  


halfAR_inv: $(OBJ_DIR)/halfAR_inv.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/hou_qr.o $(OBJ_DIR)/kernelReWY.o
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  


cglsSolver: $(OBJ_DIR)/cglsSolver.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/kernelReWY.o
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  
  
cgls_Iteration: $(OBJ_DIR)/cgls_Iteration.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/hou_qr.o $(OBJ_DIR)/kernelReWY.o  
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  
 
double_blocking_QR: $(OBJ_DIR)/double_blocking_QR.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/hou_qr.o $(OBJ_DIR)/kernelReWY.o  
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  

cgls_lowprecision: $(OBJ_DIR)/cgls_lowprecision.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/hou_qr.o $(OBJ_DIR)/kernelReWY.o  
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  

  
cgls_highprecision: $(OBJ_DIR)/cgls_highprecision.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/hou_qr.o $(OBJ_DIR)/kernelReWY.o  
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  

 
cgls_direct: $(OBJ_DIR)/cgls_direct.o $(OBJ_DIR)/utils.o $(OBJ_DIR)/matrix_generation.o $(OBJ_DIR)/qr_decomposition.o $(OBJ_DIR)/hou_qr.o $(OBJ_DIR)/kernelReWY.o  
	@mkdir -p $(BIN_DIR)  
	$(CXX) $(LFLAGS) $^ -o $(BIN_DIR)/$@  


clean:  
	rm -f $(OBJ_DIR)/*.o  
	rm -f $(BIN_DIR)/*
