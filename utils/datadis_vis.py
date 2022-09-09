import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

csv_file = "../dataset_idx/mnist/cluster_rich/50client/mnist_rich_stat.csv"

data = np.loadtxt(csv_file, delimiter=",", dtype=np.int32)
print("Num client:", data.shape[0], "Num class:", data.shape[1])

plt.rcParams["figure.figsize"] = (10,25)
plt.rcParams.update({'font.size': 16})

ax = sns.heatmap(data, annot=True, fmt="d", cbar=False, linewidths=.5)
plt.xlabel("Class")
plt.ylabel("Client")

plt.savefig("./figures/mnist_cluster_rich_N50_K10/mnist/cluster_rich/50client/mnist_rich/data_dis.png", dpi=128, box_inches="tight")
