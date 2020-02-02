import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os


def plot_single(data_path, file, experiments, iterations):
    # data preprocessing, take average
    data = np.zeros((experiments,iterations))
    with open(os.path.join(data_path, file)) as f:
        l = eval(f.readline())

        for i in range(experiments):
            data_float = np.array(l[:-1]).astype(np.float)
            data[i] = data_float
            i +=1

    data = np.average(data, axis=0)

    # plot 
    x_axis = array(range(iterations))+1

    plt.plot(x_axis, data)
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("kuhn_Poker  tsallis  alpha=1.09")

    plt.savefig("plots/kuhn_poker_tsallis_1.09.png")


def plot_stacked(data_path, files, experiments, iterations):
    # data preprocessing, take average
    data_list = []
    
    for file in files:
        print(file)
        data = np.zeros((experiments,iterations))

        with open(os.path.join(data_path, file)) as f:
            l = eval(f.readline())
            # print(l)

            for i in range(experiments):
                # data_float = np.array(l[:-1]).astype(np.float)
                data_float = np.array(l[:-1]).astype(np.float)
                data[i] = data_float
                
        data = np.average(data, axis=0)
        data_list.append(data)

    # plot 
    x_axis = array(range(iterations))+1
    alpha = 1.00
    for data in data_list:
         plt.plot(x_axis, data, label=format(alpha, '.2f') )
         alpha += 0.05
   
    # plt.plot(x_axis, data)
    plt.xlabel("Iteration")
    plt.ylabel("Exploitability")
    plt.xscale("linear")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.title("kuhn_Poker dynamics")
    
    plt.savefig(os.path.join(data_path, "plot/kuhn_poker_stacked_2.5w.png"))


if __name__ == '__main__':
    experiments = 10
    iterations = 25000

    data_path = "ICML_Experiments_Dynamics/kuhn_dynamics_2.5w/"

    files = ["kuhn_poker_dynamics_alpha_2.5w_1.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.05.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.1.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.15.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.2.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.25.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.3.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.35.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.4.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.45.txt",
            "kuhn_poker_dynamics_alpha_2.5w_1.5.txt"]

    #plot_single(data_path, "kuhn_poker_tsallis_1.09.txt", experiments, iterations)

    plot_stacked(data_path, files, experiments, iterations)

    

   