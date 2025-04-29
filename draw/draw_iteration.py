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
def plot_four_lines(datatype,list1, list2, list3, list4, label1, label2, label3, label4
                    ):
    plt.figure(figsize=(10, 8))


    if datatype in {5, 6}:
        label1 = 'standard'
        label3 = 'preconditioned'
        plt.plot(range(1, len(list1) + 1), list1, marker='o', markersize=8, label=label1)  
        plt.plot(range(1, len(list3) + 1), list3, marker='^', markersize=4, label=label3)  
    else:
        plt.plot(range(1, len(list1) + 1), list1, marker='o', markersize=8, label=label1) 
        plt.plot(range(1, len(list2) + 1), list2, marker='x', markersize=6, label=label2)
        plt.plot(range(1, len(list3) + 1), list3, marker='^', markersize=5, label=label3) 
        plt.plot(range(1, len(list4) + 1), list4, marker='.', markersize=4, label=label4) 
    distribution_type_desc = get_distribution_type_desc(datatype)

    plt.rcParams.update({'font.size': 20})
    plt.title(f'singula distribution: {distribution_type_desc}', fontsize=20, pad=20)
    plt.xlabel('iteration times', fontsize=20)
    plt.ylabel('NormsK / Norms0', fontsize=20)

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
    max_iterations = max(len(list1), len(list2), len(list3), len(list4))
    xticks = range(0, max_iterations + 1, 10)
    plt.xticks(xticks)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # plt.tick_params(axis='both', which='both', length=8, width=1.5, labelsize=25)
    # plt.legend(fontsize=25)
    plt.tick_params(axis='both', which='both', length=8, width=1.5, labelsize=20)
    plt.legend(fontsize=20)
    plt.show()


def validate_matrix_generation(data_type=1):
    base_path = "../"
    filenames = [str(data_type)+"_100_nonpre.txt", str(data_type)+"_5000_nonpre.txt", str(data_type)+"_100_pre.txt", str(data_type)+"_5000_pre.txt"]
    data_lists = []
    for filename in filenames:
        with open(base_path + filename, 'r') as file:
            data_list = [float(line.strip()) for line in file][:100]  # Limiting to 100 entries
            data_list.insert(0, 1.0)  # Insert a 1 at the beginning of the list
            data_lists.append(data_list)

            # Ensure you have exactly four lists
    datatype = data_type
    if len(data_lists) == 4:
        plot_four_lines(datatype, data_lists[0], data_lists[1], data_lists[2], data_lists[3],
                        label1='100_standard', label2='5000_standard',
                        label3='100_preconditioned', label4='5000_preconditioned')

    # Example call to function


validate_matrix_generation(6)