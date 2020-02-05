import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from numpy import array
import os
import seaborn as sns
import pandas as pd

def plot_stacked_confidence(data_arr, file, q):
    # data preprocessing
    alpha = 0
    df = pd.DataFrame(columns=["itr", "exploitability", "alpha"])

    if file == "kuhn_8000_1.txt":
        alpha = 1
    elif file == "kuhn_8000_lin_inc_1.1.txt":
        alpha = 1.1
    elif file == "kuhn_8000_lin_inc_1.2.txt":
        alpha = 1.2
    elif file == "kuhn_8000_lin_inc_1.3.txt":
        alpha = 1.3

    cur_itr = 1
    for point in data_arr:
        df = df.append({"itr" : cur_itr, "exploitability" : point, "alpha" : alpha}, ignore_index=True)
        cur_itr += 1

    q.put(df)
    return df

def listener(q,):
    '''listens for messages on the q, writes to file. '''
    df = pd.DataFrame(columns=["itr", "exploitability", "alpha"])
    while 1:
        m = q.get()
        if m == 'kill':
            sns.lineplot(x="itr", y="exploitability", hue="alpha", data=df)
            break
        df = pd.concat([df, m], ignore_index=True)
    


def main():

    experiments = 2
    iterations = 8000
    data_path = "ICML_Experiments_Dynamics/kuhn_adaptive/"

    files = ["kuhn_8000_1.txt",
             "kuhn_8000_lin_inc_1.1.txt",
             "kuhn_8000_lin_inc_1.2.txt",
             "kuhn_8000_lin_inc_1.3.txt"]
    
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
    plt.savefig(os.path.join(data_path,"plot/kuhn_stacked_ci_lin_inc.pdf"), format='pdf', dpi=2000)



if __name__ == "__main__":
  main()