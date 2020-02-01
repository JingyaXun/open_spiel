import matplotlib.pyplot as plt
import numpy as np
from numpy import array

experiments = 40

data = np.zeros((experiments,1000))

with open("leduc_poker_tsallis_1.07.txt") as f:
    l = eval(f.readline())

    for i in range(experiments):
        data_float = np.array(l[:-1]).astype(np.float)
        data[i] = data_float
        i +=1

data = np.average(data, axis=0)

# plot 
x_axis = array(range(1000))+1

plt.plot(x_axis, data)
plt.xlabel("Iteration")
plt.ylabel("Exploitability")
plt.xscale("log")
plt.yscale("log")
plt.title("leduc_Poker  tsallis  alpha=1.07")

plt.savefig("plots/leduc_poker_tsallis_1.07.png")
