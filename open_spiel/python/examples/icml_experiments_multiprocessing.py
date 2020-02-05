import multiprocessing as mp
import argparse
import subprocess
import numpy as np
from numpy import array

def experiment(args, seed, q):
    # print("args.increase: "+args.increase+"\n")
    # print("str: "+str(args.increase)+"\n")
    proc = subprocess.Popen(["python", "./neurd_example_tsallis.py", 
                "--alpha="+str(args.alpha),
                "--iterations="+str(args.iterations),
                "--game="+str(args.game),
                "--random_seed="+str(seed),
                "--adaptive_alpha="+str(args.adaptive_alpha),
                "--adaptive_policy="+str(args.adaptive_policy),
                "--increase="+str(args.increase),
                "--semi_percent="+str(args.semi_percent),
                "--exploit_rate="+str(args.exploit_rate),
                "--exp_exploit_rate="+str(args.exp_exploit_rate)], stdout=subprocess.PIPE)
    

    data = []
    results = str(proc.stdout.read())
    
    for l in results.split("\\n"):
        data.append(l.split(" ")[-1])

    q.put(data)
    return data

def hello(seed):
    print("hello " + str(seed))

def listener(q, file, args):
    '''listens for messages on the q, writes to file. '''

    with open(file, 'w') as f:
        f.write(str(args) + '\n')
        f.write("="*50) 
        f.write("\n")
        while 1:
            m = q.get()
            if m == 'kill':
                f.write('killed')
                break

            f.write(str(m) + '\n')
            f.flush()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_experiments', type=int, help="Number of experiments")
    parser.add_argument('--iterations', type=int, help="Number of iterations")
    parser.add_argument('--game', type=str, help="Name of the game")
    parser.add_argument('--alpha', type=float, help="Alpha for Tsallis")
    parser.add_argument('--adaptive_alpha', type=int, help="enable adaptive alpha")
    parser.add_argument('--adaptive_policy', type=int, help="linear=true, exponential=false")
    parser.add_argument('--increase', type=int, help="increase=true, decrease=false")
    parser.add_argument('--out', type=str, help="output file")
    parser.add_argument('--semi_percent', type=float, help="semi_percent")
    parser.add_argument('--exploit_rate', type=float, help="exploit_rate")
    parser.add_argument('--exp_exploit_rate', type=int, help="exp_exploit_rate")

    args = parser.parse_args()

    #must use Manager queue here, or will not work
    manager = mp.Manager()
    q = manager.Queue()    
    pool = mp.Pool(mp.cpu_count() + 2)

    #put listener to work first
    watcher = pool.apply_async(listener, (q,args.out,args,))

    #fire off workers
    jobs = []
    for i in range(args.num_experiments):
        job = pool.apply_async(experiment, (args, i+1, q,))
        jobs.append(job)

    # collect results from the workers through the pool result queue
    for job in jobs: 
        job.get()

    #now we are done, kill the listener
    q.put('kill')
    pool.close()
    pool.join()

    # seeds = array(range(40)) + 1

#     processes = []
#     for i in range(40):
#         p = Process(target=experiment, args=(args,i,))
#         # p = Process(target=hello, args=(i+1,))
#         processes.append(p)
#         p.start()

#     for p in processes:
#         p.join()




if __name__ == "__main__":
  main()