import pandas as pd
from matplotlib import pyplot as plt
import numpy as np



layer_list = []

def load_conflict_data(name, smooth = 10):
    data = pd.read_csv("sim_" + name + ".csv")
    #data = data.filter(layer_list)
    
    rolling_mean = data.rolling(smooth).mean()
    rolling_std = data.rolling(smooth).std()
    
    #rolling_mean *= -1 # conflict instead of similarity
    
    return rolling_mean, rolling_std



both, both_std = load_conflict_data("sarcasm_offensive")
single, single_std = load_conflict_data("hate")

plt.figure()
#plt.fill_between(range(len(both)), both["sim_1_1"] - both_std["sim_1_1"], both["sim_1_1"] + both_std["sim_1_1"], color="C0", alpha=0.2)
#plt.plot(both["sim_1_1"], label="hate - hate (trained on hate + sarcasm)", color="C0")
#plt.fill_between(range(len(both)), both["sim_1_2"] - both_std["sim_1_2"], both["sim_1_2"] + both_std["sim_1_2"], color="C2", alpha=0.2)
#plt.plot(both["sim_1_2"], label="hate - sarcasm (trained on hate + sarcasm)", color="C2")
#plt.fill_between(range(len(single)), single["sim_1_1"] - single_std["sim_1_1"], single["sim_1_1"] + single_std["sim_1_1"], color="C1", alpha=0.2)
#plt.plot(single["sim_1_1"], label="hate - hate (trained on only hate)", color="C1")
plt.fill_between(range(len(both)), both["sim_k"] - both_std["sim_k"], both["sim_k"] + both_std["sim_k"], color="C0", alpha=0.2)
plt.plot(both["sim_k"], label="hate - sarcasm: k", color="C0")
plt.fill_between(range(len(both)), both["sim_init"] - both_std["sim_init"], both["sim_init"] + both_std["sim_init"], color="C1", alpha=0.2)
plt.plot(both["sim_init"], label="hate - sarcasm: init", color="C1")
plt.fill_between(range(len(both)), both["sim_both"] - both_std["sim_both"], both["sim_both"] + both_std["sim_both"], color="C2", alpha=0.2)
plt.plot(both["sim_both"], label="hate - sarcasm: both", color="C2")

plt.ylim(-1, 1)
plt.xlim(0, len(both))
plt.title("Gradient similarity")
plt.xlabel("Episode")
plt.ylabel("Cosine similarity")
plt.grid()
plt.legend()



plt.show()