import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


matrix_sizes = [8192, 16384, 24576, 32768, 40960]
cusolver = [7.4308, 13.174, 16.8161, 18.9414, 19.7343]  
paper_method = [7.4801, 16.2876, 22.0628, 25.5444, 27.7933]


ratios = [paper / cusolver_val for paper, cusolver_val in zip(paper_method, cusolver) if cusolver_val is not None]


plt.figure(figsize=(12, 6))
plt.plot(matrix_sizes[:len(cusolver)], cusolver, marker='o', label='cuSOLVER', linewidth=2, color='orange')
plt.plot(matrix_sizes, paper_method, marker='o', label='Double Blocking', linewidth=2, color='red')

for i in range(len(ratios)):
    plt.text(matrix_sizes[i], paper_method[i], f'{ratios[i]:.2f}' + 'x', fontsize=12, ha='right', va='bottom')

plt.xlabel('matrix size', fontsize=18)
plt.ylabel('TFLOPS', fontsize=18)
plt.title('performance of QR factorization', fontsize=18)
plt.xticks(matrix_sizes, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.grid(True)


plt.show()