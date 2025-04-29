import subprocess  
import sys  
import os  

LOW_PREC_EXEC = "./bin/cgls_lowprecision"  
HIGH_PREC_EXEC = "./bin/cgls_highprecision"  

if len(sys.argv) != 7:  
    print(f"Error: Incorrect number of arguments.")  
    print(f"Usage: python {sys.argv[0]} <m> <n> <nb> <b> <datatype> <condition>")  
    sys.exit(1)  

m = sys.argv[1]  
n = sys.argv[2]  
nb = sys.argv[3]  
b = sys.argv[4]  
datatype = sys.argv[5]  
condition = sys.argv[6]  

if not os.path.exists(LOW_PREC_EXEC):  
    print(f"Error: Executable not found: {LOW_PREC_EXEC}")  
    sys.exit(1)  
if not os.path.exists(HIGH_PREC_EXEC):  
    print(f"Error: Executable not found: {HIGH_PREC_EXEC}")  
    sys.exit(1)  

cmd_low = [LOW_PREC_EXEC, m, n, nb, datatype, condition]  
print(f"Executing: {' '.join(cmd_low)}")  

try:  
    result_low = subprocess.run(cmd_low, check=False, capture_output=True, text=True)  
except FileNotFoundError:  
     print(f"Error: Failed to execute command '{LOW_PREC_EXEC}'. Check path and permissions.")  
     sys.exit(1)  
except Exception as e:  
     print(f"An unexpected error occurred during low precision execution: {e}")  
     sys.exit(1)  

print("--- Low Precision Output ---")  
print(result_low.stdout)  
if result_low.stderr:  
    print("--- Low Precision Error Output ---")  
    print(result_low.stderr)  
# print(f"--- Low Precision Exit Code (flag): {result_low.returncode} ---")  

# Check if the flag (exit code) is -1  
if result_low.returncode != 1:  
    print(f"Solution failed using the low-precision preconditioning method.")  

    cmd_high = [HIGH_PREC_EXEC, m, n, nb, b, datatype, condition]  
    print(f"Executing: {' '.join(cmd_high)}")  

    try:  
        result_high = subprocess.run(cmd_high, check=True, capture_output=True, text=True)  
        print("--- High Precision Output ---")  
        print(result_high.stdout)  
        if result_high.stderr:  
             print("--- High Precision Error Output ---")  
             print(result_high.stderr)  
        # print(f"High precision calculation completed successfully (Exit Code: {result_high.returncode}).")  

    except FileNotFoundError:  
        print(f"Error: Failed to execute command '{HIGH_PREC_EXEC}'. Check path and permissions.")  
        sys.exit(1)  
    except subprocess.CalledProcessError as e:  
        print(f"High precision calculation failed (Exit Code: {e.returncode}).")  
        print("--- High Precision Output ---")  
        print(e.stdout)  
        if e.stderr:  
            print("--- High Precision Error Output ---")  
            print(e.stderr)  
        sys.exit(e.returncode)  
    except Exception as e:  
         print(f"An unexpected error occurred during high precision execution: {e}")  
         sys.exit(1)  

elif result_low.returncode == 1:  
    print(f"Solution succeeded  using the low-precision preconditioning method.")  
else:  
    print(f"Low precision flag is {result_low.returncode}. Skipping high precision calculation as it's not -1.")  
    # Optionally exit with an error if any other flag is unexpected  
    # sys.exit(result_low.returncode)  


print("Script finished.")  
sys.exit(0)  
