import json 
import seaborn as sns
import pandas as pd
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

path_json = "dataset_idx/cifar10/dirichlet/cifar10_50_clients_dirichlet_05.json"
# path_json = "dataset_idx/mnist/dirichlet/mnist_50_clients_dirichlet_05.json"
with open(path_json, "r") as f:
    dict_json = json.load(f)

num_sample_per_class = {

}
for class_ in range(10):
    num_sample_per_class[str(class_)] = [0]*50

train_dataset = datasets.CIFAR10("benchmark/cifar10/data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]))
# train_dataset = datasets.MNIST("benchmark/mnist/data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

for client in range(50):
    list_sample_belong_client = dict_json[str(client)]
    for sample_ in list_sample_belong_client:
        image, label = train_dataset[sample_]
        num_sample_per_class[str(label)][client] += 1

# print(num_sample_per_class)

# details = {'0': [7, 88, 0, 0, 0, 0, 0, 0, 123, 0, 544, 0, 46, 0, 0, 17, 515, 14, 0, 0, 0, 0, 2271, 0, 0, 0, 5, 45, 0, 0, 0, 50, 5, 12, 11, 0, 0, 401, 0, 29, 68, 0, 749, 0, 0, 0, 0, 0, 0, 0], '1': [0, 0, 87, 0, 998, 838, 0, 207, 0, 0, 30, 0, 0, 0, 0, 197, 0, 0, 0, 2, 37, 0, 9, 0, 0, 377, 55, 0, 0, 0, 359, 0, 1381, 0, 11, 0, 0, 10, 2, 0, 0, 105, 0, 251, 9, 0, 19, 2, 1, 13], '2': [0, 3, 0, 0, 3, 78, 0, 17, 0, 34, 0, 0, 0, 3, 145, 10, 146, 0, 0, 625, 1565, 714, 2, 0, 0, 3, 0, 664, 108, 4, 0, 4, 0, 555, 18, 1, 0, 0, 1, 2, 0, 0, 130, 0, 0, 0, 0, 127, 0, 38], '3': [0, 672, 0, 30, 1, 0, 38, 1068, 45, 0, 0, 2, 599, 0, 10, 0, 0, 0, 4, 0, 0, 209, 110, 0, 20, 226, 356, 0, 0, 0, 16, 13, 0, 0, 65, 10, 0, 0, 159, 0, 1, 2, 5, 0, 4, 5, 0, 0, 3, 1327], '4': [470, 28, 0, 1, 0, 674, 0, 0, 0, 1138, 21, 0, 0, 616, 0, 5, 1, 0, 0, 1, 514, 4, 0, 0, 0, 0, 0, 1, 3, 52, 0, 13, 0, 0, 0, 196, 16, 0, 29, 0, 5, 0, 0, 103, 180, 73, 2, 21, 0, 833], '5': [13, 1, 0, 801, 68, 0, 1, 0, 0, 0, 6, 355, 0, 27, 233, 3, 20, 11, 1018, 0, 9, 0, 373, 376, 2, 3, 36, 1, 18, 0, 26, 0, 0, 29, 7, 342, 124, 686, 0, 20, 2, 25, 0, 0, 243, 119, 2, 0, 0, 0], '6': [8, 0, 0, 0, 0, 0, 22, 10, 0, 4, 0, 960, 2100, 0, 1, 25, 65, 0, 0, 25, 0, 502, 1, 1, 0, 0, 0, 0, 131, 31, 0, 0, 0, 0, 0, 6, 127, 0, 0, 1, 0, 0, 0, 11, 7, 0, 0, 50, 18, 894], '7': [93, 48, 0, 135, 0, 0, 1, 3, 2, 247, 0, 0, 0, 0, 29, 13, 51, 0, 0, 118, 0, 2, 16, 0, 0, 0, 0, 489, 60, 0, 837, 199, 181, 55, 185, 0, 0, 0, 74, 0, 4, 0, 213, 0, 53, 0, 1491, 0, 21, 380], '8': [405, 0, 2, 797, 34, 57, 0, 0, 103, 778, 10, 0, 0, 87, 0, 1, 2, 248, 107, 0, 1, 22, 2, 58, 10, 1, 158, 0, 5, 0, 70, 1076, 232, 0, 1, 0, 0, 505, 0, 0, 0, 15, 40, 162, 0, 0, 0, 1, 0, 10], '9': [109, 50, 0, 0, 0, 1, 39, 0, 0, 0, 17, 359, 0, 5, 0, 920, 0, 0, 8, 0, 0, 0, 484, 0, 101, 26, 10, 8, 772, 1415, 27, 0, 0, 0, 9, 0, 25, 0, 0, 0, 0, 0, 0, 2, 0, 0, 16, 323, 210, 64]}

df = pd.DataFrame.from_dict(num_sample_per_class, orient = 'index')

df[df == 0] = np.nan

fig, ax = plt.subplots(figsize=(50,10))
colors = [(1, 1, 1), (0.2, 0.4, 0.8)]  # White to Blue
cmap = LinearSegmentedColormap.from_list("Custom", colors)
ax = sns.heatmap(df, annot=True, cmap=cmap, fmt=".1f", linewidths=.5, mask=df.isnull())

# plt.savefig("mnist_dir_05.png")
plt.savefig("cifar10_dir_05.png")
print(df)