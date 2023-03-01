import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

path = 'dataset_idx/cifar10/noniid/CIFAR10_30client_pareto_15_attacker_1000sample.csv'
data = pd.read_csv(path, header=None)
sns.set(rc={'figure.figsize':(8,12)})
sns_figure = sns.heatmap(data, annot=True, fmt=".0f")
sns_figure.set(xlabel ="Class", ylabel = "Client", title ='Non iid pareto')
# sns_figure.set_yticks(range(50))
# sns_figure.set_xticklabels(['0','a','b','c','d','e'])
fig = sns_figure.get_figure()
fig.savefig('noniid_pareto_CIFAR10_30client_pareto_15_attacker_1000sample.png')