import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plot_updates_components(model, updates, peers_types, epoch, proportion, attacked_class, dirty_rate, num_malicious, agg_algorithm, algorithm):
    weight=False
    bias=False
    for i in range(3):
        if i ==1 :
            weight=True
        elif i == 2:
            weight=False
            bias=True
        flattened_updates, sum_grads = flatten_updates(model, updates, weight=weight, bias=bias, attacked_class=attacked_class)
        # flattened_updates = StandardScaler().fit_transform(flattened_updates)
        # # if i !=2:
        # pca = PCA(n_components=2)
        # principalComponents = pca.fit_transform(flattened_updates)
        # principalDf = pd.DataFrame(data = principalComponents,
        #                             columns = ['c1', 'c2'])
        # if i ==2:
        #     principalDf = pd.DataFrame(data = flattened_updates,
        #                                 columns = ['c1', 'c2'])
        
        sum_gradsDf = pd.DataFrame(data = sum_grads,
                                    columns = ['sum_grads'])
        peers_typesDf = pd.DataFrame(data = peers_types,
                                    columns = ['target'])
        # finalDf = pd.concat([principalDf, sum_gradsDf, peers_typesDf['target']], axis = 1)
        # if i==1:
        grad_per_neuron = pd.DataFrame(data = flattened_updates,
                                        columns = ['o0','o1','o2','o3','o4','o5','o6','o7','o8','o9'])
        finalDf = pd.concat([grad_per_neuron, sum_gradsDf, peers_typesDf['target']], axis = 1)
        path_csv = algorithm + '/' + agg_algorithm + '/' 'attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/csv/{}'.format(len(attacked_class),dirty_rate,proportion*50,num_malicious,i)
        if not os.path.exists(path_csv):
            os.makedirs(path_csv)
        finalDf.to_csv(path_csv + '/' + 'epoch{}.csv'.format(epoch))
        
        # fig = plt.figure(figsize = (7,7))
        # ax = fig.add_subplot(1,1,1) 
        # ax.set_xlabel('Component 1', fontsize = 10)
        # ax.set_ylabel('Component 2', fontsize = 10)
        # ax.set_title('2 component PCA', fontsize = 15)
        # targets = ['benign', 'attacker']
        # colors = ['green', 'red']
        # for target, color in zip(targets,colors):
        #     indicesToKeep = finalDf['target'] == target
        #     ax.scatter(finalDf.loc[indicesToKeep, 'c1'], 
        #                 finalDf.loc[indicesToKeep, 'c2'], 
        #                 c = color, 
        #                 edgecolors='gray',
        #                 s = 80)
        # # ax.set_xlim([-5, 5])
        # # ax.set_ylim([-5, 5])
        # ax.legend(targets)
        # path_pca = 'grad/attacked_class_{}/dirty_rate_{}/proportion_{}/num_malicious_{}/pca/{}'.format(len(attacked_class),dirty_rate,proportion*50,num_malicious,i)
        # # path_pca = 'grad/{}/{}/pca/{}'.format(len(attacked_class),proportion*50,i)
        # if not os.path.exists(path_pca):
        #     os.makedirs(path_pca)
        # plt.savefig(path_pca + '/' + 'epoch{}.png'.format(epoch))
        # # plt.show()


def flatten_updates(model, updates, weight=False, bias=False, attacked_class=None):
    # breakpoint()
    flatten_updates = []
    
    if weight == True:
        for idx, update in enumerate(updates):
            
            # output_grad = 0
            for i, (name, params) in enumerate(model.named_parameters()):
                        if params.requires_grad:
                            if name == 'linear.weight':
                                weight_grad = torch.sum(update[i],dim=1) 
                            # if name =='linear.bias':
                            #     weight_grad = update[i]
            flatten_updates.append(weight_grad.cpu().detach().numpy())
        return np.array(flatten_updates), np.sum(np.array(flatten_updates),axis=1)
        
    elif bias ==True:
        # for idx, update in enumerate(updates):
           
        #     bias_grad = []
            # # for class_ in attacked_class:
            #     # x=0
            #     for i, (name, params) in enumerate(model.named_parameters()):
            #                 if params.requires_grad:
            #                     # if name == 'linear.weight':
            #                     #     x = torch.sum(update[i][class_]).item()
                                    
                                    
            #                     if name =='linear.bias':
            #                         x = update[i][class_].item()
                                    
        #     bias_grad.append(x)
            
        #     flatten_updates.append(bias_grad)
        # return np.array(flatten_updates), np.sum(np.array(flatten_updates),axis=1)
        for idx, update in enumerate(updates):
            
            # output_grad = 0
            for i, (name, params) in enumerate(model.named_parameters()):
                        if params.requires_grad:
                            # if name == 'linear.weight':
                            #     weight_grad = torch.sum(update[i],dim=1) 
                            if name =='linear.bias':
                                bias_grad = update[i]
            flatten_updates.append(bias_grad.cpu().detach().numpy())
        return np.array(flatten_updates), np.sum(np.array(flatten_updates),axis=1)
        
        
    # for idx, update in enumerate(updates):
    #     flatten_updates.append([])
    #     for layer in range(len(update)):
    #         flatten_updates[idx].append(torch.sum(update[layer]).item())
    # return np.array(flatten_updates), np.sum(np.array(flatten_updates),-1)
    for idx, update in enumerate(updates):
            
            # output_grad = 0
            for i, (name, params) in enumerate(model.named_parameters()):
                        if params.requires_grad:
                            if name == 'linear.weight':
                                weight_bias_grad = torch.sum(update[i],dim=1) 
                            if name =='linear.bias':
                                weight_bias_grad += update[i]
            flatten_updates.append(weight_bias_grad.cpu().detach().numpy())
    return np.array(flatten_updates), np.sum(np.array(flatten_updates),axis=1)
        
