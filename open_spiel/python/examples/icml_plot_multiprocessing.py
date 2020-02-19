import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
import seaborn as sns
import pandas as pd

def plot_stacked_confidence(data_arr, file, q):
    # data preprocessing
    beta = 0
    df = pd.DataFrame(columns=["itr", "exploitability", "beta"])

    if file == "kuhn_poker_0_1_lin_1_10000_0.5_100_1_2.txt":
        beta = 1
    elif file == "kuhn_poker_1_1_lin_1.1_10000_0.5_100_1_2_seed.txt":
        beta = 1.1
    elif file == "kuhn_poker_1_1_lin_1.2_10000_0.5_100_1_2_seed.txt":
        beta = 1.2
    elif file == "kuhn_poker_1_1_lin_1.3_10000_0.5_100_1_2_seed.txt":
        beta = 1.3
    elif file == "kuhn_poker_1_1_lin_1.4_10000_0.5_100_1_2_seed.txt":
        beta = 1.4

    cur_itr = 1
    for point in data_arr:
        df = df.append({"itr" : cur_itr, "exploitability" : point, "beta" : beta}, ignore_index=True)
        cur_itr += 1
    q.put(df)
    return df

def listener(q,):
    '''listens for messages on the q, writes to file. '''
    df = pd.DataFrame(columns=["itr", "exploitability", "beta"])
    while 1:
        m = q.get()
        print("in listener")
        print(m)
        if m == 'kill':
            print("killing listener")
            print(m)
            sns.lineplot(x="itr", y="exploitability", hue="beta", data=df)
            break
        df = pd.concat([df, m], ignore_index=True)
        print(df)
    


def main():

    experiments = 5
    iterations = 100
    data_path = "ICML_Experiments_Final/kuhn_poker_players_experiments/2_players_seed_1-10/"

    files = ["kuhn_poker_0_1_lin_1_10000_0.5_100_1_2.txt",
             "kuhn_poker_1_1_lin_1.1_10000_0.5_100_1_2_seed.txt",
             "kuhn_poker_1_1_lin_1.2_10000_0.5_100_1_2_seed.txt",
             "kuhn_poker_1_1_lin_1.3_10000_0.5_100_1_2_seed.txt",
             "kuhn_poker_1_1_lin_1.4_10000_0.5_100_1_2_seed.txt"]
    
    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,))

    #fire off workers
    jobs = []
    for file in files:
        with open(os.path.join(data_path, file)) as f:
            l = f.readline()
            l = f.readline()
            for i in range(experiments):
                l = eval(f.readline())
                data_float = np.array(l[:iterations]).astype(np.float)

                job = pool.apply_async(plot_stacked_confidence, (data_float, file, q,))
                jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    plt.xlabel("Iteration")
    plt.ylabel("NashConv")
    plt.xscale("linear")
    plt.yscale("log")
    plt.tight_layout()
    # plt.margins(0.08,0.08)
    plt.savefig(os.path.join(data_path,"plot/stacked_test.pdf"), format='pdf', dpi=2000)



if __name__ == "__main__":
  main()