import sys  
import os  
import subprocess  
import shlex # For safe command string splitting  

# --- Configuration ---  
CGLS_DIRECT_EXEC = "./bin/cgls_direct"  
CGLS_LOW_EXEC = "./bin/cgls_lowprecision"  
CGLS_HIGH_EXEC = "./bin/cgls_highprecision"  
PREDICT_SCRIPT = "LSTM/prediction_script.py" # Assuming predict.py is in the same directory or PATH  

# --- Helper Function to Run Commands ---  
def run_command(cmd_list, check=True, capture=False):  
    print(f"Executing: {' '.join(shlex.quote(str(c)) for c in cmd_list)}")  
    try:  
        result = subprocess.run(cmd_list, check=check, capture_output=capture, text=True, errors='ignore')  
        if capture:  
            print("--- Command Output ---")  
            print(result.stdout)  
            if result.stderr:  
                print("--- Command Error Output ---")  
                print(result.stderr)  
            print("---------------------")  
        return result  
    except FileNotFoundError:  
        print(f"Error: Command not found: {cmd_list[0]}. Check path and permissions.")  
        sys.exit(1)  
    except subprocess.CalledProcessError as e:  
        print(f"Error: Command failed with exit code {e.returncode}: {' '.join(shlex.quote(str(c)) for c in cmd_list)}")  
        if capture:  
            print("--- Failing Command Output ---")  
            print(e.stdout)  
            if e.stderr:  
                print("--- Failing Command Error Output ---")  
                print(e.stderr)  
            print("---------------------------")  
        sys.exit(1)  
    except Exception as e:  
        print(f"An unexpected error occurred while running {' '.join(shlex.quote(str(c)) for c in cmd_list)}: {e}")  
        sys.exit(1)  

# --- Argument Parsing ---  
if len(sys.argv) != 7:  
    print(f"Error: Incorrect number of arguments.")  
    print(f"Usage: python {sys.argv[0]} <m> <n> <nb> <b> <datatype> <condition>")  
    sys.exit(1)  

m = sys.argv[1]  
n = sys.argv[2]  
nb = sys.argv[3]  
b_arg = sys.argv[4] # 'b' argument needed only for high precision  
datatype = sys.argv[5]  
condition = sys.argv[6]  

# --- Check for executables ---  
required_execs = [CGLS_DIRECT_EXEC, CGLS_LOW_EXEC, CGLS_HIGH_EXEC, PREDICT_SCRIPT]  
for exe in required_execs:  
    if not os.path.exists(exe):  
         # predict.py might be found via PATH, so os.path.exists might not be enough  
         # A better check for python script might be shutil.which, but let's keep it simple  
         # If predict.py isn't directly runnable, subprocess will fail later anyway.  
         if exe == PREDICT_SCRIPT and not os.access(exe, os.X_OK):  
             print(f"Warning: {PREDICT_SCRIPT} not found or not executable in current directory. Assuming it's in PATH.")  
         elif exe != PREDICT_SCRIPT:  
             print(f"Error: Required file not found: {exe}")  
             sys.exit(1)  


# --- Step 1: Run initial cgls_direct ---  
initial_iterations = "5"  
cmd_direct_initial = [CGLS_DIRECT_EXEC, m, n, nb, datatype, condition, initial_iterations]  
run_command(cmd_direct_initial, check=True, capture=False) # Don't capture, assume output goes to file  

# --- Step 2: Determine results filename ---  
results_filename = f"res{datatype}_{condition}_nonpre.txt"  
print(f"Attempting to read features from: {results_filename}")  

if not os.path.exists(results_filename):  
    print(f"Error: Results file not found: {results_filename}")  
    sys.exit(1)  

features_for_predict = []  
file_features = []  

print(f"Attempting to read 5 features from: {results_filename}")  

if not os.path.exists(results_filename):  
    print(f"Error: Results file not found: {results_filename}")  
    sys.exit(1)  

try:  
    with open(results_filename, 'r') as f:  
        line = f.readline()  
        if line:  
            # Read the values from the file  
            file_features = [float(val) for val in line.strip().split()]  
        else:  
            print(f"Error: Results file '{results_filename}' is empty.")  
            sys.exit(1)  

    # Check if exactly 5 features were read from the file  
    if len(file_features) != 5:  
        print(f"Error: Expected 5 features in '{results_filename}', but found {len(file_features)}.")  
        sys.exit(1)  

    # Construct the final 7 features: m, n, followed by the 5 from the file  
    # Ensure m and n are also converted to float  
    features_for_predict = [int(m), int(n)] + file_features  

    print(f"Successfully read 5 features from file. Prepared 7 total features for prediction: {features_for_predict}")  

except FileNotFoundError:  
     # This might be redundant due to the check above, but good practice  
     print(f"Error: Results file not found after check: {results_filename}")  
     sys.exit(1)  
except ValueError as e:  
     print(f"Error: Could not convert value to float. Check m ('{m}'), n ('{n}'), or file content in '{results_filename}'. Details: {e}")  
     sys.exit(1)  
except Exception as e:  
     print(f"An error occurred while reading '{results_filename}' or preparing features: {e}")  
     sys.exit(1)  

# --- Step 4: Run predict.py ---  
# Ensure the number of features passed matches predict.py's expectation (now 7)  
cmd_predict = ['python', PREDICT_SCRIPT] + [str(f) for f in features_for_predict]  
predict_result = run_command(cmd_predict, check=True, capture=True) # Execute and capture output 
# --- Step 5: Parse predict.py output for the flag ---  
predicted_flag = None  
try:  
    for line in predict_result.stdout.splitlines():  
        line_stripped = line.strip()  
        if line_stripped.startswith("Predicted class (flag):"):  
            predicted_flag = int(line_stripped.split(":")[-1].strip())  
            print(f"Detected prediction flag: {predicted_flag}")  
            break # Found the flag  
    if predicted_flag is None:  
        print(f"Error: Could not find 'Predicted class (flag):' line in the output of {PREDICT_SCRIPT}.")  
        sys.exit(1)  
except ValueError:  
    print(f"Error: Could not convert the predicted flag value to an integer.")  
    sys.exit(1)  
except Exception as e:  
    print(f"An error occurred while parsing the output of {PREDICT_SCRIPT}: {e}")  
    sys.exit(1)  


# --- Step 6: Execute final command based on flag ---  
final_cmd = None  
if predicted_flag == 1:  
    final_iterations = "50"  
    final_cmd = [CGLS_DIRECT_EXEC, m, n, nb, datatype, condition, final_iterations]  
elif predicted_flag == 2:  
    final_cmd = [CGLS_LOW_EXEC, m, n, nb, datatype, condition]  
elif predicted_flag == 3:  
    final_cmd = [CGLS_HIGH_EXEC, m, n, nb, b_arg, datatype, condition]  
else:  
    print(f"Error: Unknown prediction flag value: {predicted_flag}. Expected 1, 2, or 3.")  
    sys.exit(1)  

print(f"Flag is {predicted_flag}, executing the corresponding final command.")  
run_command(final_cmd, check=True, capture=False) # Run the selected final command  

print("Script finished successfully.")  
sys.exit(0)  
