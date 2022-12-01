from noniid_featured import noniid_featured
from noniid_pareto import noniid_pareto
from noniid_quantitative import noniid_quantitative
from torchvision import datasets
from torchvision import datasets, transforms
import json
import os
from pathlib import Path

def save_dataset_idx(list_idx_sample, path, filename="dataset_idx.json"):
    if not Path(path).exists():
        os.system(f"mkdir -p {path}")
        
    with open(path + filename, "w+") as outfile:
        json.dump(list_idx_sample, outfile)

import pandas as pd

def sta(client_dict, train_dataset, num_client=10, num_label=10):
    rs = []
    for client in range(num_client):
        tmp = []
        for i in range(num_label):
            tmp.append(sum(train_dataset[j][1] == i for j in client_dict[client]))
        rs.append(tmp)
    df = pd.DataFrame(rs,columns=[f"Label_{i}" for i in range(num_label)])
    return df
  
def gen_full(dataset,total_client, num_sample=50000):
    client_dict = {}
    for i in range(1):
        if i ==0:
            client_dict[i] = range(num_sample)
        else:
            client_dict[i] = []
        client_dict[i] = [int(k) for k in client_dict[i]]
    return client_dict


if __name__ == '__main__':
    # train_dataset = datasets.CIFAR10("../../benchmark/cifar10/data/", train=True, download=False, transform=None)
    train_dataset = datasets.EMNIST("../../benchmark/emnist/data/", split='letters', train=True, download=True, transform=None)
    print("total sample of dataset", len(train_dataset))
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=['featured', 'quantitative', 'pareto'])
    parser.add_argument("--total_client", type=int, default=100, required=False)
    args = parser.parse_args()
    
    if args.method == 'featured':
        client_dict = noniid_featured(train_dataset, total_client=args.total_client, total_label=26)
    elif args.method == 'quantitative':
        client_dict = noniid_quantitative(train_dataset, total_client=args.total_client, total_label=26)
    elif args.method == 'pareto':
        client_dict = noniid_pareto(train_dataset, total_client=args.total_client, total_label=26)
    else:
        raise Exception("Not recognise data distributing method")
    
    if client_dict is None: 
        print("Pareto unsuccessful")
        exit(0)
        
    print("Gen done!")
    save_dataset_idx(client_dict, f"../../dataset_idx/emnist/{args.total_client}client/", f"emnist_{args.total_client}client_{args.method}.json")
    df = sta(client_dict, train_dataset, num_client=args.total_client, num_label=10)
    print("Total sample in modified dataset", df.values.sum())
    df.to_csv(f"../../dataset_idx/emnist/{args.total_client}client/emnist_{args.total_client}client_{args.method}.csv", index=False, header=False)
  
    