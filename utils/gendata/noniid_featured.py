import numpy as np
import math

def noniid_featured(dataset, total_client, total_label=100):
    label_list = [i for i in range(total_label)]
    label_per_client = 2
    total_sample = len(dataset)
    
    labels = dataset.targets
    idxs = range(total_sample)
    idxs_labels = np.vstack((idxs, labels)).T
    
    dict_client = {}
    client_labels = []
    for _ in range(math.ceil(total_label/label_per_client)):
        this_set = np.random.choice(label_list, 2, replace=False)
        client_labels.append(list(this_set))
        label_list = list(set(label_list) - set(this_set))
    
    num_added = (total_client - len(client_labels))
    client_labels = client_labels + [client_labels[-1]] * num_added
    
    sample_per_client = int(np.floor(total_sample/total_label * 1/(num_added + 1)))
    
    for client_idx, client_label in zip(range(total_client), client_labels):
        label_1, label_2 = client_label
        
        idxes_1 = idxs_labels[idxs_labels[:,1] == label_1][:,0]
        idxes_2 = idxs_labels[idxs_labels[:,1] == label_2][:,0]

        # print(idxes_1.shape, idxes_2.shape)
        
        label_1_idxes = np.random.choice(idxes_1, sample_per_client, replace=False)
        label_2_idxes = np.random.choice(idxes_2, sample_per_client, replace=False)
        
        dict_client[client_idx] = label_1_idxes.tolist()
        dict_client[client_idx] += label_2_idxes.tolist()
            
        idxs_labels[label_1_idxes] -= 100
        idxs_labels[label_2_idxes] -= 100
    
    return dict_client