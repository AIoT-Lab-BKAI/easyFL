import torch
from torch import nn
from utils.fmodule import get_module_from_model, _modeldict_sum
import copy


@torch.no_grad()
def special_aggregate(last_layer_list):
    P, Q = 0, 0
    for last_layer in last_layer_list:
        alpha = torch.norm(last_layer, dim=1, keepdim=True) * torch.mean((last_layer > 0) * 1.0, dim=1, keepdim=True)
        P += alpha * last_layer
        Q += alpha
    
    return torch.nan_to_num(P/Q, 0)


def model_sum(ms):
    if not ms: return None
    op_with_graph = sum([mi.ingraph for mi in ms]) > 0
    res = copy.deepcopy(ms[0]).to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = modeldict_sum(mpks, special_aggregate)
            for l in mlr[n]._parameters.keys():
                if mlr[n]._parameters[l] is None: continue
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        modeldict_cp(res.state_dict(), _modeldict_sum([mi.state_dict() for mi in ms]))
    return res


def modeldict_sum(mds, special_aggregate_method=None):
    if not mds: return None
    md_sum = {}
    
    keys = list(mds[0].keys())
    for layer in mds[0].keys():
        md_sum[layer] = torch.zeros_like(mds[0][layer])
    
    last_layers = []
    
    for wid in range(len(mds)):
        for layer in keys:
            if mds[0][layer] is None:
                md_sum[layer] = None
                continue
            
            if layer == keys[-2] and special_aggregate_method is not None:
                last_layers.append(mds[wid][layer])
            else:
                md_sum[layer] = md_sum[layer] + mds[wid][layer]
    
    if len(last_layers):
        md_sum[keys[-2]] = special_aggregate_method(last_layers)
       
    return md_sum

def modeldict_cp(md1, md2):
    for layer in md1.keys():
        md1[layer].data.copy_(md2[layer])
    return

def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res