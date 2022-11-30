import numpy as np
import math

def cifar_noniid_quantitative(dataset, total_client, ratio=0.6, total_sample=50000, total_label=10):
    group_mem = math.ceil(ratio * total_client)
    labels = dataset.targets
    idxs = range(total_sample)

    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    labels = idxs_labels[1, :]
    tmp = 0
    list_tmp = []
    for i in range(total_sample):
        if(dataset[idxs[i]][1] == tmp):
            tmp +=1
            list_tmp.append(i)
    list_tmp.append(total_sample)

    list_dict = [0] * total_label
    for i in range(total_label):
        list_dict[i] = idxs[list_tmp[i]:list_tmp[i+1]]

    dict_client = {}
    n_samples= int(5000 / group_mem)
    for i in range(total_client):
        dict_client[i] = []
    # client_dis = [1]*total_label
    client_dis = np.random.normal(1, 0.2, total_client)

    index_list = [i for i in range(total_label)]

    label_client = np.random.choice(index_list, int(total_label * 0.2), replace=False)
    for i in range(group_mem):
        # label_client = range(int(total_label * 0.2))
        for j in label_client:
            a = np.random.choice(list_dict[j], int(n_samples*client_dis[i]), replace=False)
            list_dict[j] = list(set(list_dict[j]) - set(a))
            dict_client[i] = dict_client[i]  + list(a)
        dict_client[i] = [int(k) for k in dict_client[i]]

    index_list = list(set(index_list) - set(label_client))
    for i in range(group_mem,total_client,1):
        # label_client = np.random.choice(range(int(total_label * 0.2), total_label), int(total_label * 0.2),replace=False)
        print("here list", index_list)
        label_client = np.random.choice(index_list, int(total_label * 0.2), replace=False)
        for j in label_client:
            a = np.random.choice(list_dict[j], math.ceil(n_samples*client_dis[i]), replace=False)
            list_dict[j] = list(set(list_dict[j]) - set(a))
            dict_client[i] = dict_client[i]  + list(a)
        dict_client[i] = [int(k) for k in dict_client[i]]
        index_list = list(set(index_list) - set(label_client))
        
    return dict_client



def cifar100_noniid_quantitative(dataset, total_client, total_label=100):
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
    adds_client_label = [client_labels[-1]] * num_added
    adds_client_Nsample = np.round(np.random.default_rng().normal(0.65/(num_added + 1), 0.1/(num_added + 1), num_added + 1) * (total_sample/total_label))
    client_Nsample = np.round(np.random.default_rng().normal(0.5, 0.1, len(client_labels) - 1) * (total_sample/total_label))
    
    client_labels = client_labels + adds_client_label
    client_Nsample = client_Nsample.tolist() + adds_client_Nsample.tolist()
    
    for client_idx, client_label, sample_per_client in zip(range(total_client), client_labels, client_Nsample):
        label_1, label_2 = client_label
        
        idxes_1 = idxs_labels[idxs_labels[:,1] == label_1][:,0]
        idxes_2 = idxs_labels[idxs_labels[:,1] == label_2][:,0]

        # print(idxes_1.shape, idxes_2.shape)
        
        label_1_idxes = np.random.choice(idxes_1, int(sample_per_client), replace=False)
        label_2_idxes = np.random.choice(idxes_2, int(sample_per_client), replace=False)
        
        dict_client[client_idx] = label_1_idxes.tolist()
        dict_client[client_idx] += label_2_idxes.tolist()
            
        idxs_labels[label_1_idxes] -= 100
        idxs_labels[label_2_idxes] -= 100
        
    return dict_client


from torchvision import datasets

if __name__ == "__main__":
    train_dataset = datasets.CIFAR100( "./data/cifar/", train=True, download=False, transform=None)
    dict_return = cifar100_noniid_quantitative(train_dataset, 100)