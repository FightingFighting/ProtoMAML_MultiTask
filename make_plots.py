import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



def plot_similarity(emotions):
    xs = [i * 10 for i in range(len(data[emotions]["encoder"]))]
    plt.plot(xs, data[emotions]["encoder"], label=emotions)



data = pd.read_json("similarity_meta_init.json")

plt.figure()

for emotions in data.keys():
    plot_similarity(emotions)

plt.ylim(-1, 1)

plt.legend()
plt.grid()

plt.show()