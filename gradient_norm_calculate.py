import csv
import numpy as np
import pandas as pd
from numpy import linalg as LA

path_ = 'mp_fedavg/pareto_fedavg/attacked_class_10/dirty_rate_0.0/proportion_10.0/num_malicious_0/csv/0/'
gradient_norm = []
for round in range(1, 201):
    path_round = path_ + f'epoch{round}.csv'
    with open(path_round, 'r') as f:
        gradient_round = pd.read_csv(f, index_col=0)
        for i in range(10):
            gradient_norm_i = gradient_round.iloc[i, 0:10]
            # print(gradient_norm_i)
            gradient_norm.append(LA.norm(gradient_norm_i))
            # print(gradient_norm)
