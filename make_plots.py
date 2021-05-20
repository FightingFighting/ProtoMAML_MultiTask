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



inner, stds = load_data("gradient_similarities/sarcasm_offensive/sarcasm_offensive_outer.csv")



layers = []
layer_stds = []
labels = []
for i in range(12):
    layer_name = "encoder.encoder.layer." + str(i) + ".attention.output.dense.weight"
    layers.append(inner[layer_name])
    layer_stds.append(stds[layer_name])
    labels.append(layer_name)



plt.figure()

for i, param in enumerate(layers):
    #plt.fill_between(range(len(param)), param - layer_stds[i], param + layer_stds[i], alpha = 0.2)
    plt.plot(param, label = labels[i])

plt.grid()
plt.legend()



'''plt.figure()

means = []
xs = []
colors = []
labels = []
i = 0
for param in inner:
    #plt.plot(inner[param], label=param)
    try:
        index = int(param.split(".")[3])
    except:
        continue
    mean = inner[param].mean()
    means.append(mean)
    #labels.append(".".join(param.split(".")[4:]))
    labels.append(str(i))
    xs.append(index + i/20)
    colors.append('C' + str(i))
    i += 1
    if i == 16: i = 0

plt.bar(xs, means, width=1/20, label = labels)

plt.grid()
plt.legend()'''

plt.figure()

means = {}
for param in inner:
    try:
        layer = int(param.split(".")[3])
    except:
        continue
    type = ".".join(param.split(".")[4:])
    if type not in means: means[type] = [0] * 12
    means[type][layer] = inner[param].mean()

for type in means:
    plt.plot(means[type], label=type)

plt.grid()
plt.legend()



plt.show()