import numpy as np
import matplotlib.pyplot as plt

# Data preparation
data_1000 = {
    "Judgment Time": 27.364,
    "QR Time1": 60.139,
    "Iteration Time1": 58.257,
    "QR Time3": 0,
    "Iteration Time3": 0,
    "cusolver Time": 149.916,
    "QR Time2": 61.139,
    "Iteration Time2": 60.325,
    "High Precision QR Time": 0,
    "Calculation Time": 0,
}

data_10000 = {
    "Judgment Time": 36.353,
    "QR Time1": 152,
    "Iteration Time1": 177,
    "QR Time3": 0,
    "Iteration Time3": 0,
    "cusolver Time": 423.777,
    "QR Time2": 152.312,
    "Iteration Time2": 186.252,
    "High Precision QR Time": 0,
    "Calculation Time": 0,
}

data_100000 = {
    "Judgment Time": 84.369,
    "QR Time1": 585,
    "Iteration Time1": 719,
    "QR Time3": 0,
    "Iteration Time3": 0,
    "cusolver Time": 1528,
    "QR Time2": 585.58,
    "Iteration Time2": 719.111,
    "High Precision QR Time": 0,
    "Calculation Time": 0,
}


# Extract and organize all data
matrix_conditions = [data_1000, data_10000, data_100000]

# Extract each time component
judgment_time = []
qr_time1 = []
iteration_time1 = []
qr_time3 = []
iteration_time3 = []
cusolver_time = []
qr_time2 = []
iteration_time2 = []
high_precision_qr_time = []
calculation_time = []

for data in matrix_conditions:
    judgment_time.append(data['Judgment Time'])
    qr_time1.append(data['QR Time1'])
    iteration_time1.append(data['Iteration Time1'])
    qr_time3.append(data['QR Time3'])
    iteration_time3.append(data['Iteration Time3'])
    cusolver_time.append(data['cusolver Time'])
    qr_time2.append(data['QR Time2'])
    iteration_time2.append(data['Iteration Time2'])
    high_precision_qr_time.append(data['High Precision QR Time'])
    calculation_time.append(data['Calculation Time'])

# Convert to Numpy arrays
judgment_time = np.array(judgment_time)
qr_time1 = np.array(qr_time1)
iteration_time1 = np.array(iteration_time1)
qr_time3 = np.array(qr_time3)
iteration_time3 = np.array(iteration_time3)
cusolver_time = np.array(cusolver_time)
qr_time2 = np.array(qr_time2)
iteration_time2 = np.array(iteration_time2)
high_precision_qr_time = np.array(high_precision_qr_time)
calculation_time = np.array(calculation_time)

# Set labels and bar positions
labels = ["8192", "16384", "32768"]
x = np.arange(len(labels))  # Positions for the labels
width = 0.25  # Width of each bar

# Create subplots with square aspect ratio
fig, ax = plt.subplots(figsize=(8, 8))  # Set figsize to be square

# Plot stacked bar chart
# Paper method (Judgment Time + QR Time1 + Iteration Time1)
rects1 = ax.bar(x - width, judgment_time, width, label='Judgment Time', color='lightblue')
rects2 = ax.bar(x - width, qr_time1, width, bottom=judgment_time, label='Low precision QR Time', color='lightgreen', alpha = 0.5)
rects3 = ax.bar(x - width, iteration_time1, width, bottom=judgment_time + qr_time1, label='Low precision Iteration Time', color='salmon', alpha = 0.7)
rects4 = ax.bar(x - width, qr_time3, width, bottom=judgment_time + qr_time1 + iteration_time1, label='High precision QR Time', color='pink', alpha = 0.5) # 添加
rects5 = ax.bar(x - width, iteration_time3, width, bottom=judgment_time + qr_time1 + iteration_time1 + qr_time3, label='Calculation Time', color='gray', alpha = 0.5) # 添加

# CUDASolver Time
rects6 = ax.bar(x, cusolver_time, width, label='cuSOLVER Time', color='orange', alpha = 0.7)

# Comparison method (QR Time2 + Iteration Time2 + High Precision + Calculation)
rects7 = ax.bar(x + width, qr_time2, width, color='lightgreen', alpha = 0.5)
rects8 = ax.bar(x + width, iteration_time2, width, bottom=qr_time2, color='salmon', alpha = 0.7)
rects9 = ax.bar(x + width, high_precision_qr_time, width, bottom=qr_time2 + iteration_time2, color='pink', alpha = 0.5)
rects10 = ax.bar(x + width, calculation_time, width, bottom=qr_time2 + iteration_time2 + high_precision_qr_time, color='gray', alpha = 0.5)

# Calculate ratios
ratios_cusolver = cusolver_time / (judgment_time + qr_time1 + iteration_time1 + qr_time3 + iteration_time3)  # Ratio for cuSOLVER
ratios_comparison = (qr_time2 + iteration_time2 + high_precision_qr_time + calculation_time) / (judgment_time + qr_time1 + iteration_time1 + qr_time3 + iteration_time3)  # Ratio for comparison

# Annotate ratios above the appropriate bars
for i in range(len(labels)):
    ax.annotate(f'{ratios_cusolver[i]:.2f}x', xy=(i, cusolver_time[i]), xytext=(0, 3),
                textcoords="offset points", ha='center', fontsize=12, color='black')

    ax.annotate(f'{ratios_comparison[i]:.2f}x', xy=(i + width, qr_time2[i] + iteration_time2[i] + high_precision_qr_time[i] + calculation_time[i]),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12, color='black')

# Chart setup
ax.set_ylabel('Time (ms)', fontsize=18)
ax.set_title('normal', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.set_xlabel('Matrix Size', fontsize=18)
ax.legend(fontsize=12)

# Adjust layout automatically
fig.tight_layout()

# Show the plot
plt.show()