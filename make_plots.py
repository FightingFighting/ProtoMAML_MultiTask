import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



def load_conflict_data(name, smooth = 100):
    data = pd.read_csv("sim_" + name + ".csv")
    
    rolling_mean = data.rolling(smooth).mean()
    rolling_std = data.rolling(smooth).std()
    
    #rolling_mean *= -1 # conflict instead of similarity
    
    return rolling_mean, rolling_std



both, both_std = load_conflict_data("sarcasm_hate")

plt.figure()
plt.fill_between(range(len(both)), both["sim_k"] - both_std["sim_k"], both["sim_k"] + both_std["sim_k"], color="C0", alpha=0.2)
plt.plot(both["sim_k"], label="k", color="C0")
plt.fill_between(range(len(both)), both["sim_init"] - both_std["sim_init"], both["sim_init"] + both_std["sim_init"], color="C1", alpha=0.2)
plt.plot(both["sim_init"], label="init", color="C1")
plt.fill_between(range(len(both)), both["sim_both"] - both_std["sim_both"], both["sim_both"] + both_std["sim_both"], color="C2", alpha=0.2)
plt.plot(both["sim_both"], label="both", color="C2")

plt.fill_between(range(len(both)), both["loss"] - both_std["loss"], both["loss"] + both_std["loss"], color="C3", alpha=0.2)
plt.plot(both["loss"], label="loss", color="C3")

plt.ylim(-1, 1)
plt.xlim(0, len(both))
plt.title("Gradient similarity")
plt.xlabel("Episode")
plt.ylabel("Cosine similarity")
plt.grid()
plt.legend()



plt.show()