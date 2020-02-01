import argparse
import subprocess


# exec(open("neurd_example_tsallis.py").read())

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_experiments', type=int, help="Number of experiments")
    parser.add_argument('--iterations', type=int, help="Number of iterations")
    parser.add_argument('--game', type=str, help="Name of the game")
    parser.add_argument('--alpha', type=float, help="Alpha for Tsallis")
    parser.add_argument('--out', type=str, help="output file")

    args = parser.parse_args()

    with open(str(args.out), 'w') as f:
        f.write("game:"+args.game+"\n")
        f.write("iterations:"+str(args.iterations)+"\n")
        f.write("alpha:"+str(args.alpha)+"\n")
        f.write("num_experiments:"+str(args.num_experiments)+"\n")
        f.write("\n")

        for i in range(args.num_experiments):
            # f.write("experiment_num:"+str(i+1)+"\n")

            proc = subprocess.Popen(["python", "./neurd_example_tsallis.py", 
            "--alpha="+str(args.alpha),
            "--iterations="+str(args.iterations),
            "--game="+str(args.game),
            "--random_seed="+str(i+1)], stdout=subprocess.PIPE)

            data = []
            results = str(proc.stdout.read())

            for l in results.split("\\n"):
                data.append(l.split(" ")[-1])

            f.write(str(data)+"\n")


if __name__ == "__main__":
  main()