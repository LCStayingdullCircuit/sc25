import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def get_distribution_type_desc(distribution_type):
    distribution_map = {
        1: "cluster_1",
        2: "cluster_0",
        3: "geometric",
        4: "arithmetic",
        5: "normal",
        6: "uniform"
    }
    return distribution_map.get(distribution_type, "Unknown")

def plot_single_line(data, label, distribution_type_desc):
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(data) + 1), data, marker='o', label=label)


    plt.rcParams.update({'font.size': 20})
    plt.title(f'singular distribution: {distribution_type_desc}', fontsize=20, pad=20)
    plt.xlabel('index of singular value', fontsize=20)
    plt.ylabel('singular', fontsize=20)

    class LargeNumberFormatter(FuncFormatter):
        def __init__(self, large_num_format='{x:.1f}'):
            self.large_num_format = large_num_format
            FuncFormatter.__init__(self, self.func)

        def func(self, x, pos):
            return self.large_num_format.format(x=x)

    formatter = LargeNumberFormatter()
    plt.gca().yaxis.set_major_formatter(formatter)

    # plt.xticks(fontsize=25)
    # plt.yticks(fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.tick_params(axis='both', which='both', length=8, width=1.5, labelsize=25)
    # plt.legend(fontsize=25)
    plt.tick_params(axis='both', which='both', length=8, width=1.5, labelsize=20)
    plt.legend(fontsize=20)
    plt.show()

def validate_matrix_generation(data_type=1):
    base_path = "../"
    filename = f"matrixRES2_10000_{data_type}.txt"
    full_path = base_path + filename

    data_list = []
    with open(full_path, 'r') as file:
        data_list = [float(line.strip()) for line in file]  # Remove the limit of 100 entries and the insertion of 1.0

    distribution_type_desc = get_distribution_type_desc(data_type)
    # plot_single_line(data_list, label=f'matrixRES2_10000_{data_type}', distribution_type_desc=distribution_type_desc)
    plot_single_line(data_list, label='_nolegend_',distribution_type_desc=distribution_type_desc)

validate_matrix_generation(6)