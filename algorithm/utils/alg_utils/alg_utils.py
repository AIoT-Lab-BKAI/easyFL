import numpy as np
import torch
import copy


def generate_branches(best_ans):
    branches = [] # list of possible answer
    # Adding to existing groups
    for idx in range(1, len(best_ans)):
        new_ans = copy.deepcopy(best_ans)
        new_ans[idx].append(best_ans[0][0])
        new_ans[0].pop(0)
        branches.append(new_ans)

    # Create a new group
    new_ans = copy.deepcopy(best_ans) # [[2,3,4,5,6], [1]]
    new_ans.append([new_ans[0].pop(0)]) # [[3,4,5,6], [1], [2]]
    branches.append(new_ans)
    return branches

def cluster_norm(weights, cluster):
    # cluster = [1, 2, 3, 4, 5, 6]
    s = 0
    for idx in cluster:
        s += weights[idx]
    # return np.power(2 * len(cluster), 0.5) * torch.norm(s).item()
    return np.power(len(cluster), 1) * torch.norm(s).item()**2


def eval_ans(weights, ans):
    # ans = [[1, 2, 3, 4, 5, 6], [0]]
    s = 0
    for cluster in ans:
        if len(cluster):
            s += cluster_norm(weights, cluster)
        else:
            return None
    # print(ans, "scores", s)
    return s

def eval_branch(weights, branches):
    # ans is the index
    eval_list = []
    for ans in branches:
        eval_list.append(eval_ans(weights, ans))
    return eval_list


def dummy_DFS_clustering(weights):
    # print("Weights in", weights.shape)
    best_ans = [[i for i in range(weights.shape[0])]]
    best_score = eval_ans(weights, best_ans)
    while True:
        branches = generate_branches(best_ans)
        eval_list = eval_branch(weights, branches)
        
        if None in eval_list:
            # print("Exists an empty cluster! Exit")
            break
        
        best_ans_idx = np.argmin(eval_list)
        
        if eval_list[best_ans_idx] > best_score:
            print("Stop at ans", best_ans, "found no better solutions, scores", best_score)
            break
        else:
            print("Found better ans", branches[best_ans_idx], "scores", eval_list[best_ans_idx])
            best_ans = branches[best_ans_idx]
            best_score = eval_list[best_ans_idx]
            
    return best_ans, best_score