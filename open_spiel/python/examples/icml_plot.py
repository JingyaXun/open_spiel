import matplotlib.pyplot as plt
import numpy as np
from numpy import array

experiments = 40

data = np.zeros((experiments,1000))

with open("kuhn_poker_tsallis_1.00.txt") as f:
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
plt.title("Kuhn_Poker  tsallis  alpha=1.00")

plt.savefig("plots/kuhn_poker_tsallis_1.00.png")
