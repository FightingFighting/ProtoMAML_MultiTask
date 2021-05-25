import json
import matplotlib.pyplot as plt
import pandas as pd



with open('similarity.json', 'r') as f:
    data = json.load(f)



for ind, (task_pair, similarity) in enumerate(data.items()):
    if "sarcasm-" in task_pair:
        y = pd.DataFrame(similarity["all_model"])
        plt.plot(y[0].rolling(20).mean(),label=task_pair)
plt.legend()
plt.show()



for ind, (task_pair, similarity) in enumerate(data.items()):
    if "sarcasm-" in task_pair:
        y = pd.DataFrame(similarity["fc_layer"])
        plt.plot(y[0].rolling(20).mean(),label=task_pair)
plt.legend()
plt.show()



for ind, (task_pair, similarity) in enumerate(data.items()):
    if "sarcasm-" in task_pair:
        y = pd.DataFrame(similarity["encoder"])
        plt.plot(y[0].rolling(20).mean(),label=task_pair)
plt.legend()
plt.show()