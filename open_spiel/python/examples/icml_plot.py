import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
import seaborn as sns
import pandas as pd


sns.set()

font = {'family' : 'normal',
        'size'   : 35}


plt.rc('font', **font)

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
                data_float = np.array(l[:iterations]).astype(np.float)
                data[i] = data_float

        # keep top n results (based on the last value in the tensor)
        # ind=np.argsort(data[:,-1])
        # data[5:,:]

        # data_sorted = []
        # for i in range(5):
        #     data_sorted.append(data[ind[i], :])
        # data = np.array(data_sorted)
        data = np.average(data, axis=0)
        data_list.append(data)

    # plot 
    x_axis = array(range(iterations))+1
    alpha = 1.0
    for data in data_list:
        if alpha == 1.0:
            plt.plot(x_axis, data, label="NeuRD" )
        else:
            plt.plot(x_axis, data, label="GeNeuRD, "+r'$\beta$'+"="+format(alpha, '.2f') )
        alpha += 0.05
   
    # plt.plot(x_axis, data)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("NashConv", fontsize=14)
    plt.xscale("linear")
    plt.yscale("log")
    plt.legend(loc="upper right")
    # plt.title("Leduc_Poker Dynamics")
    
    # plt.savefig(os.path.join(data_path, "kuhn_3w_lin_inc.png"))

    plt.tight_layout()
    plt.margins(0.08,0.08)
    plt.savefig(os.path.join(data_path,"leduc_2player_GeNeuRD.pdf"), format='pdf', dpi=2000)

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
    beta = 1.0

    for file in files:
        print(file)
        data = np.zeros((experiments,iterations))
        with open(os.path.join(data_path, file)) as f:
            l = f.readline()
            l = f.readline()
            for i in range(experiments):
                l = eval(f.readline())
                data_float = np.array(l[:iterations]).astype(np.float)
                data[i] = data_float

            # keep top n results (based on the last value in the tensor)
            ind=np.argsort(data[:,-1])
            data[5:,:]
            # data_sorted = []
            # for i in range(5):
            #     data_sorted.append(data[ind[i], :])
            # data = np.array(data_sorted)

            for experiment in data:
                cur_itr = 1
                for exp in experiment:
                    df = df.append({"itr" : cur_itr, "exploitability" : exp, "beta" : format(beta, '.2f')}, ignore_index=True)
                    cur_itr += 1

        beta += 0.05

    # plot 
    # colors = ["windows blue", "orange", "faded green", "dusty purple"]
    colors = ["windows blue", "orange", "faded green", "dusty purple", "salmon pink"]

    # new_labels = ['NeuRD', 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.05", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.10", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.15"]
    # new_labels = ['NeuRD', 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.10", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.20", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.30", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.40"]
    new_labels = ['NeuRD', 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.05", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.10", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.15", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.20"]
    # new_labels = ['NeuRD', 'GeNeuRD, '+r'$\beta$'+"=1.10", 'GeNeuRD, '+r'$\beta$'+"=1.20", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.10", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.20"]
    # new_labels = ['NeuRD', 'GeNeuRD, '+r'$\beta$'+"=1.20", 'GeNeuRD, '+r'$\beta$'+"=1.40", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.20", 'AdaGeNeuRD, '+r'$\beta_T$'+"=1.40"]
    
    g = sns.lineplot(x="itr", y="exploitability", hue="beta", palette=sns.xkcd_palette(colors), data=df)
    # legend = g._legend
    # legend.set_title("")
    # for t, l in zip(legend.texts, new_labels):
    #     t.set_text(l)
    plt.legend(new_labels)
    plt.xlabel("Iteration")
    plt.ylabel("NashConv")
    plt.xscale("linear")
    plt.yscale("log")
    plt.tight_layout()
    plt.margins(0.08,0.08)
    plt.savefig(os.path.join(data_path,"kuhn_ci_3p_2w_last5_second.pdf"), format='pdf', dpi=2000)


def plot_lines (data_path, file, experiments, iterations):
    # data preprocessing, take average
    
    data_list = np.zeros((experiments,iterations))

    with open(os.path.join(data_path, file)) as f:
        l = f.readline()
        l = f.readline()
        for i in range(experiments):
            l = eval(f.readline())

            # data_float = np.array(l[:-1]).astype(np.float)
            data_float = np.array(l[:iterations]).astype(np.float)
            data_list[i] = data_float
    
            
    # plot 
    x_axis = array(range(iterations))+1
    alpha = 1.0
    for d in data_list:
        if alpha == 1.0:
            plt.plot(x_axis, d, label="NeuRD" )
        else:
            plt.plot(x_axis, d, label="AdaGeNeuRD, "+r'$\beta$'+"="+format(alpha, '.2f') )
        alpha += 0.1
   
    # plt.plot(x_axis, data)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("NashConv", fontsize=14)
    plt.xscale("linear")
    plt.yscale("log")
    plt.legend(loc="upper right")
    # plt.title("Leduc_Poker Dynamics")
    
    # plt.savefig(os.path.join(data_path, "plot/leduc_poker_stacked_10000_dynamics.png"))

    plt.tight_layout()
    plt.margins(0.08,0.08)
    plt.savefig(os.path.join(data_path,"kuhn_lin_inc_13_all_seeds.pdf"), format='pdf', dpi=2000)

if __name__ == '__main__':
    experiments = 5
    iterations = 50000

    # data_path = "ICML_Experiments_Appendix/kuhn_poker_players_experiments/"
    # data_path = "ICML_Experiments_Final/kuhn_poker_players_experiments/2_players_seed_1-10/"
    data_path = "ICML_Experiments_Rebattle/leduc_poker_players_experiments/"

    # files = ["kuhn_poker_0_1_lin_1_30000_0.5_100_1_4.txt",
    #          "kuhn_poker_1_1_lin_1.05_30000_0.5_100_1_4.txt",
    #          "kuhn_poker_1_1_lin_1.1_30000_0.5_100_1_4.txt",
    #          "kuhn_poker_1_1_lin_1.15_30000_0.5_100_1_4.txt"]

    # files = ["kuhn_poker_0_1_lin_1_10000_0.5_100_1_2.txt",
    #          "kuhn_poker_1_1_lin_1.1_10000_0.5_100_1_2_seed.txt",
    #          "kuhn_poker_1_1_lin_1.2_10000_0.5_100_1_2_seed.txt",
    #          "kuhn_poker_1_1_lin_1.3_10000_0.5_100_1_2_seed.txt",
    #          "kuhn_poker_1_1_lin_1.4_10000_0.5_100_1_2_seed.txt"]

    # files = ["kuhn_poker_0_1_lin_1_40000_0.5_100_1_3.txt",
    #         "kuhn_poker_0_1_lin_1.1_40000_0.5_100_1_3.txt",
    #         "kuhn_poker_0_1_lin_1.2_40000_0.5_100_1_3.txt",
    #         "kuhn_poker_1_1_lin_1.1_20000_0.5_100_1_3.txt",
    #         "kuhn_poker_1_1_lin_1.2_20000_0.5_100_1_3.txt"]

    
    files = ["leduc_poker_0_1_lin_1_50000_0.5_100_1_2.txt",
            "leduc_poker_0_1_lin_1.05_50000_0.5_100_1_2.txt",
            "leduc_poker_0_1_lin_1.1_50000_0.5_100_1_2.txt",
            "leduc_poker_0_1_lin_1.15_50000_0.5_100_1_2.txt",
            "leduc_poker_0_1_lin_1.2_50000_0.5_100_1_2.txt"]

    # plot_single(data_path, "kuhn_poker_tsallis_1.09.txt", experiments, iterations)

    plot_stacked(data_path, files, experiments, iterations)
    
    # plot_lines(data_path, file, experiments, iterations)

    # plot_stacked_confidence(data_path, files, experiments, iterations)
    

    