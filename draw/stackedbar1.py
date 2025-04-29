import numpy as np
import matplotlib.pyplot as plt

# Data preparation
data_1000 = {
    "Judgment Time": 18.352,
    "QR Time1": 0,
    "Iteration Time1": 29.09,
    "cusolver Time": 430.894,
    "QR Time2": 152.571,
    "Iteration Time2": 79.184,
    "High Precision QR Time": 0,
    "Calculation Time": 0,
}

data_10000 = {
    "Judgment Time": 18.356,
    "QR Time1": 0,
    "Iteration Time1": 29.064,
    "cusolver Time": 565.334,
    "QR Time2": 151.312,
    "Iteration Time2": 192.475,
    "High Precision QR Time": 0,
    "Calculation Time": 0,
}

data_100000 = {
    "Judgment Time": 18.324,
    "QR Time1": 0,
    "Iteration Time1": 29.023,
    "cusolver Time": 1230.14,
    "QR Time2": 152.312,
    "Iteration Time2": 285.475,
    "High Precision QR Time": 0,
    "Calculation Time": 0,
}

# Extract and organize all data
matrix_conditions = [data_1000, data_10000, data_100000]

# Extract each time component
judgment_time = []
qr_time1 = []
iteration_time1 = []
cusolver_time = []
qr_time2 = []
iteration_time2 = []
high_precision_qr_time = []
calculation_time = []

for data in matrix_conditions:
    judgment_time.append(data['Judgment Time'])
    qr_time1.append(data['QR Time1'])
    iteration_time1.append(data['Iteration Time1'])
    cusolver_time.append(data['cusolver Time'])
    qr_time2.append(data['QR Time2'])
    iteration_time2.append(data['Iteration Time2'])
    high_precision_qr_time.append(data['High Precision QR Time'])
    calculation_time.append(data['Calculation Time'])

# Convert to Numpy arrays
judgment_time = np.array(judgment_time)
qr_time1 = np.array(qr_time1)
iteration_time1 = np.array(iteration_time1)
cusolver_time = np.array(cusolver_time)
qr_time2 = np.array(qr_time2)
iteration_time2 = np.array(iteration_time2)
high_precision_qr_time = np.array(high_precision_qr_time)
calculation_time = np.array(calculation_time)

# Set labels and bar positions
labels = ["1000", "10000", "100000"]
x = np.arange(len(labels))  # Positions for the labels
width = 0.25  # Width of each bar

# Create subplots with square aspect ratio
fig, ax = plt.subplots(figsize=(8, 8))  # Set figsize to be square

# Plot stacked bar chart
# Paper method (Judgment Time + QR Time1 + Iteration Time1)
rects1 = ax.bar(x - width, judgment_time, width, label='Judgment Time', color='lightblue', alpha=0.5)
# rects2 = ax.bar(x - width, qr_time1, width, bottom=judgment_time, label='QR Time1', color='lightgreen')
rects3 = ax.bar(x - width, iteration_time1, width, bottom=judgment_time + qr_time1, label='cgls Time', color='salmon', alpha=0.7)

# CUDASolver Time
rects4 = ax.bar(x, cusolver_time, width, label='cuSOLVER Time', color='orange', alpha=0.7)

# Comparison method (QR Time2 + Iteration Time2 + High Precision + Calculation)
rects5 = ax.bar(x + width, qr_time2, width, label='Low precision QR Time', color='lightgreen', alpha=0.5)
rects6 = ax.bar(x + width, iteration_time2, width, bottom=qr_time2, label='Iteration Time', color='salmon', alpha=0.7)
rects7 = ax.bar(x + width, high_precision_qr_time, width, bottom=qr_time2 + iteration_time2, label='High Precision QR Time', color='pink', alpha=0.5)
rects8 = ax.bar(x + width, calculation_time, width, bottom=qr_time2 + iteration_time2 + high_precision_qr_time, label='Calculation Time', color='gray', alpha=0.5)

# Calculate ratios
ratios_cusolver = cusolver_time / (judgment_time + qr_time1 + iteration_time1)  # Ratio for cuSOLVER
ratios_comparison = (qr_time2 + iteration_time2 + high_precision_qr_time + calculation_time) / (judgment_time + qr_time1 + iteration_time1)  # Ratio for comparison

# Annotate ratios above the appropriate bars
for i in range(len(labels)):
    ax.annotate(f'{ratios_cusolver[i]:.1f}x', xy=(i, cusolver_time[i]), xytext=(0, 3),
                textcoords="offset points", ha='center', fontsize=12, color='black')

    ax.annotate(f'{ratios_comparison[i]:.1f}x', xy=(i + width, qr_time2[i] + iteration_time2[i] + high_precision_qr_time[i] + calculation_time[i]),
                xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12, color='black')

# Chart setup
ax.set_ylabel('Time (ms)', fontsize=18)
ax.set_title('Cluster 1', fontsize=18)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=16)
ax.set_xlabel('Condition number', fontsize=18)
ax.legend(fontsize=12)

# Adjust layout automatically
fig.tight_layout()

# Show the plot
plt.show()