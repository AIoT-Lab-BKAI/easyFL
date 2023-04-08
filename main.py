from builtins import breakpoint
import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing
import wandb

class MyLogger(flw.Logger):
    def __init__(self):
        super().__init__()
        
    def log(self, server=None):
        if server==None: return
        if self.output == {}:
            self.output = {
                "meta":server.option,
                "mean_curve":[],
                "var_curve":[],
                "train_losses":[],
                "test_accs":[],
                "test_losses":[],
                "valid_accs":[],
                "client_accs":{},
                "mean_valid_accs":[],
                "inference_time": []
            }
        if "mp_" in server.name:
            test_metric, test_loss, inference_time = server.test(device=torch.device('cuda'))
        else:
            test_metric, test_loss, inference_time = server.test(device="cuda")
        
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid', 'cuda')
        # train_metrics, train_losses = server.test_on_clients(self.current_round, 'train', 'cuda')
        train_metrics, train_losses = (valid_metrics, valid_losses)
        
        self.output['train_losses'].append(1.0*sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)])/server.data_vol)
        self.output['valid_accs'].append(valid_metrics)
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        self.output['mean_valid_accs'].append(1.0*sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)])/server.data_vol)
        self.output['mean_curve'].append(np.mean(valid_metrics))
        self.output['var_curve'].append(np.std(valid_metrics))
        self.output['inference_time'].append(inference_time)
        
        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name]=[self.output['valid_accs'][i][cid] for i in range(len(self.output['valid_accs']))]
        
        print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
        print(self.temp.format("Validating Accuracy:", self.output['mean_valid_accs'][-1]))
        print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
        print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))
        print(self.temp.format("Mean of Inference Time:", self.output['inference_time'][-1]))
        
        # wandb record
        if server.wandb:
            wandb.log(
                {
                    "Training Loss":        self.output['train_losses'][-1], 
                    "Testing Loss":         self.output['test_losses'][-1],
                    "Testing Accuracy":     self.output['test_accs'][-1],
                    "Validating Accuracy":  self.output['mean_valid_accs'][-1],
                    "Mean Client Accuracy": self.output['mean_curve'][-1],
                    "Std Client Accuracy":  self.output['var_curve'][-1],
                    "Inference Time":       self.output['inference_time'][-1],
                    "Max Testing Accuracy": max(self.output['test_accs'])
                }
            )


logger = MyLogger()

def main():
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)   # number of gpus
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    
    if option['wandb']:
        wandb.init(
            project="dirtyFL", 
            entity="aiotlab",
            group=f"{option['task']}_{option['noise_type']}",
            name=f"num_malicious_{option['num_malicious']}_dirty_rate_{option['dirty_rate'][0]}_attacked_class_{len(option['attacked_class'])}_{option['agg_algorithm']}_beta_{option['ours_beta']}",
            config=option
        )
        
        wandb.define_metric("Testing Accuracy", summary="max")
        wandb.define_metric("Inference Time", summary="mean")
    
    print("CONFIG =>", option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




