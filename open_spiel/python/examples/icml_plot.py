import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
import seaborn as sns
import pandas as pd


sns.set()


def plot_single(data_path, file, experiments, iterations):
    # data preprocessing, take average
    data = np.zeros((experiments,iterations))
    with open(os.path.join(data_path, file)) as f:

        for i in range(experiments):
            l = eval(f.readline())
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
    plt.title("Leduc_Poker ")

    plt.savefig("plots/kuhn_poker_tsallis_1.09.png")


def plot_stacked(data_path, files, experiments, iterations):
    # data preprocessing, take average
    data_list = []
    
    for file in files:
        print(file)
        data = np.zeros((experiments,iterations))

        with open(os.path.join(data_path, file)) as f:
            l = f.readline()
            l = f.readline()
            for i in range(experiments):
                l = eval(f.readline())
                
                # data_float = np.array(l[:-1]).astype(np.float)
                data_float = np.array(l[:iterations]).astype(np.float)
                data[i] = data_float
                
        data = np.average(data, axis=0)
        data_list.append(data)

    # plot 
    x_axis = array(range(iterations))+1
    alpha = 1.0
    for data in data_list:
         plt.plot(x_axis, data, label=format(alpha, '.2f') )
         alpha += 0.1
   
    # plt.plot(x_axis, data)
    plt.xlabel("Iteration")
    plt.ylabel("NashConv")
    plt.xscale("linear")
    plt.yscale("log")
    plt.legend(loc="upper right")
    # plt.title("Leduc_Poker Dynamics")
    
    # plt.savefig(os.path.join(data_path, "plot/leduc_poker_stacked_10000_dynamics.png"))

    plt.tight_layout()
    plt.margins(0.08,0.08)
    plt.savefig(os.path.join(data_path,"kuhn_fixed.pdf"), format='pdf', dpi=2000)

def plot_single_confidence(data_path, file, experiments, iterations):
    # data preprocessing, take average
    data = np.zeros((experiments,iterations))
    with open(os.path.join(data_path, file)) as f:

        for i in range(experiments):
            l = eval(f.readline())
            
            data_float = np.array(l[:iterations]).astype(np.float)
            data[i] = data_float
            i +=1
    
    # convert matrix to dataframe object
    cur_itr = 1

    df = pd.DataFrame(columns=["itr", "exploitability"])

    for line in data:
        cur_itr = 1
        for exp in line:
            df = df.append({"itr" : cur_itr, "exploitability" : exp}, ignore_index=True)
            cur_itr += 1

    # plot 
    x_axis = array(range(iterations))+1

    # plt.plot(x_axis, data)
    # plt.xlabel("Iteration")
    # plt.ylabel("Exploitability")
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.title("kuhn_Poker  tsallis  alpha=1.09")

    p = sns.lineplot(x="itr", y="exploitability", hue="alpha", data=df)

    plt.savefig(os.path.join(data_path, "plot/kuhn_poker_ci_1.09.png"))


def plot_stacked_confidence(data_path, file, experiments, iterations):
    # data preprocessing
    df = pd.DataFrame(columns=["itr", "exploitability", "beta"])
    alpha = 1.0
    for file in files:
        print(file)
        with open(os.path.join(data_path, file)) as f:
            l = f.readline()
            l = f.readline()
            for i in range(experiments):
                l = eval(f.readline())
                data_float = np.array(l[:iterations]).astype(np.float)
                cur_itr = 1
                for exp in data_float:
                    df = df.append({"itr" : cur_itr, "exploitability" : exp, "beta" : format(alpha, '.2f')}, ignore_index=True)
                    cur_itr += 1
        alpha += 0.1

    # plot 
    colors = ["windows blue", "orange", "faded green", "dusty purple"]

    sns.lineplot(x="itr", y="exploitability", hue="beta", palette=sns.xkcd_palette(colors), data=df)
    plt.xlabel("Iteration")
    plt.ylabel("NashConv")
    plt.xscale("linear")
    plt.yscale("log")
    plt.tight_layout()
    plt.margins(0.08,0.08)
    plt.savefig(os.path.join(data_path,"plot/kuhn_stacked_ci_lin_dec.pdf"), format='pdf', dpi=2000)


if __name__ == '__main__':
    experiments = 10
    iterations = 8000

    data_path = "ICML_Experiments_Dynamics/kuhn_adaptive/"

    # data_path = "ICML_Experiments_Dynamics/leduc_poker_experiments/"

    # data_path = "ICML_Experiments_Dynamics/"

    # file = "kuhn_poker_dynamics_alpha_2.5w_1.1.txt"

    # files=[
    #     "leduc_poker_1_1_exp_1.1_10000.txt",
    #     "leduc_poker_1_1_exp_1.2_10000.txt",
    #     "leduc_poker_1_1_exp_1.3_10000.txt",
    #     "leduc_poker_1_1_exp_1.4_10000.txt",
    #     "leduc_poker_1_1_exp_1.5_10000.txt"]

    # files = [
    #          "kuhn_poker_dynamics_alpha_1w_1.txt",
    #          "kuhn_poker_dynamics_alpha_1.1.txt",
    #          "kuhn_poker_dynamics_alpha_1w_1.2.txt",
    #          "kuhn_poker_dynamics_alpha_1w_1.3.txt"
    #          ]

    files = ["kuhn_8000_1.txt",
             "kuhn_8000_lin_dec_1.1.txt",
             "kuhn_8000_lin_dec_1.2.txt",
             "kuhn_8000_lin_dec_1.3.txt"]

    # files = ["kuhn_poker_dynamics_alpha_1w_1.txt",
    #          "kuhn_poker_dynamics_alpha_1w_1.1.txt",
    #          "kuhn_poker_dynamics_alpha_1w_1.2.txt",
    #          "kuhn_poker_dynamics_alpha_1w_1.3.txt"]

    #plot_single(data_path, "kuhn_poker_tsallis_1.09.txt", experiments, iterations)

    #plot_stacked(data_path, files, experiments, iterations)

    plot_stacked_confidence(data_path, files, experiments, iterations)
    

   