import numpy as np
import math

def noniid_featured(dataset, total_client, total_label=100):
    label_per_client = 7
    total_sample = len(dataset)
    
    labels = dataset.targets
    label_list = [i for i in np.unique(labels)]
    idxs = range(total_sample)
    idxs_labels = np.vstack((idxs, labels)).T
    
    dict_client = {client_idx: [] for client_idx in range(total_client)}
    client_labels = []
    for _ in range(math.ceil(total_label/label_per_client)):
        if len(label_list) > label_per_client:
            this_set = np.random.choice(label_list, label_per_client, replace=False)
        else:
            this_set = label_list.copy()
            
        client_labels.append(list(this_set))
        label_list = list(set(label_list) - set(this_set))
    
    num_added = (total_client - len(client_labels))
    x1 = int(num_added * 0.4)
    x2 = int(num_added * 0.3)
    x3 = num_added - x1 - x2
    adds_client_label = [client_labels[-1]] * x1 + [client_labels[-2]] * x2 + [client_labels[-3]] * x3
    client_labels = client_labels + adds_client_label
    
    sample_per_client = int(np.floor(total_sample/total_label * 1/(num_added + 1)))
    
    for client_idx, client_label in zip(range(total_client), client_labels):
        for label in client_label:
            idxes = idxs_labels[idxs_labels[:,1] == label][:,0]
            label_idxes = np.random.choice(idxes, sample_per_client, replace=False)
            dict_client[client_idx] += label_idxes.tolist()
            idxs_labels[label_idxes] -= 100
    
    return dict_client