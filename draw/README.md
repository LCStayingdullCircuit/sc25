## Scripts for Plotting Figures in the Paper

### 1. `draw_iteration.py`
- **Purpose**: Plots the CGLS and low-precision preconditioned CGLS, corresponding to Figure 1 in the paper.
- **Usage**:  
  - **Prerequisite**: Run `cgls_Iteration.cu` first.
  - **Matrix Selection**: Change the argument of `validate_matrix_generation()` to select the matrix distribution to be executed.

### 2. `draw_singular.py`
- **Purpose**: Plots the singular value distribution of AR-1, corresponding to Figure 2 in the paper.
- **Usage**:  
  - **Prerequisite**: Run `halfAR_inv.cu` first.
  - **Matrix Selection**: Likewise, adjust the argument in `validate_matrix_generation()` to select the matrix distribution.

### 3. `qr.py`
- **Purpose**: Plots the performance of the QR decomposition, corresponding to Figure 5 in the paper.
- **Usage**:  
  - **Prerequisite**: Run `double_blocking_QR.cu` beforehand.
  - **Data Update**: Update the data source in `qr.py` as required.
  - **Parameter Configuration**: Select matrix size, block size, matrix distribution, and condition number via console arguments.

### 4. `stackedbar*.py`
- **Purpose**: Plots the performance of mixed-lls solvers, corresponding to Figure 6 in the paper.
- **Usage**:  
  - **Prerequisites**: Run `runcgls_mixedLLS.py`, `runcgls_original.py`, and `cglsSolver.cu` in advance.
    - These correspond to the lls solver proposed in the paper, the conventional method solver, and the cuSOLVER solver respectively.
  - **Data Modification**: Data files such as `data_1000`, `data_10000`, and `data_100000` correspond to condition numbers 1,000, 10,000, and 100,000 respectively.

### Parameter Explanation

- **Judgment Time**: Time for decision process in the proposed method.
- **QR Time1**: Low-precision QR time in the proposed method.
- **Iteration Time1**: Low-precision iteration time in the proposed method.
- **cusolver Time**: Solver time using cuSOLVER.
- **QR Time2**: Low-precision QR time of comparative method.
- **Iteration Time2**: Low-precision iteration time of comparative method.
- **High Precision QR Time**: High-precision QR decomposition time.
- **Calculation Time**: High-precision solver time.


