from multiprocessing import Process
import argparse
import subprocess
import numpy as np
from numpy import array

def experiment(args, seed):
    with open(str(args.out), 'w+') as f:
        proc = subprocess.Popen(["python", "./neurd_example_tsallis.py", 
                "--alpha="+str(args.alpha),
                "--iterations="+str(args.iterations),
                "--game="+str(args.game),
                "--random_seed="+str(seed)], stdout=subprocess.PIPE)

        data = []
        results = str(proc.stdout.read())

        for l in results.split("\\n"):
            data.append(l.split(" ")[-1])

        f.write(str(data)+"\n")

def hello(seed):
    print("hello " + str(seed))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_experiments', type=int, help="Number of experiments")
    parser.add_argument('--iterations', type=int, help="Number of iterations")
    parser.add_argument('--game', type=str, help="Name of the game")
    parser.add_argument('--alpha', type=float, help="Alpha for Tsallis")
    parser.add_argument('--out', type=str, help="output file")

    args = parser.parse_args()

    # seeds = array(range(40)) + 1

    for i in range(40):
        p = Process(target=experiment, args=(args,i,))
        # p = Process(target=hello, args=(i+1,))
        p.start()
        p.join()


if __name__ == "__main__":
  main()