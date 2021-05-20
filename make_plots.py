import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



layer_list = []

def load_data(path, smooth = 10):
    data = pd.read_csv(path)
    #data = data.filter(layer_list)
    
    rolling_mean = data.rolling(smooth).mean()
    rolling_std = data.rolling(smooth).std()
    
    return rolling_mean, rolling_std



inner, stds = load_data("sim_sarcasm_offensive.csv")



plt.figure()
plt.fill_between(range(len(inner)), inner["sim_1_1"] - stds["sim_1_1"], inner["sim_1_1"] + stds["sim_1_1"], color="C1", alpha=0.2)
plt.plot(inner["sim_1_1"], label="Same task", color="C1")
plt.fill_between(range(len(inner)), inner["sim_1_2"] - stds["sim_1_2"], inner["sim_1_2"] + stds["sim_1_2"], color="C2", alpha=0.2)
plt.plot(inner["sim_1_2"], label="Different task", color="C2")
plt.grid()
plt.legend()
plt.show()