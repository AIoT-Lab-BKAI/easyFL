import utils.fflow as flw
import numpy as np
import torch
import os
import multiprocessing
import wandb

class MyLogger(flw.Logger):
    def __init__(self):
        super().__init__()
        self.max_acc = 0
        
    def log(self, server=None, round=None):
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
                "mean_valid_accs":[]
            }
        
        # test_metric, test_loss = server.test(device="cuda")
        valid_metrics, _ = server.test_on_clients(dataflag='valid', device='cuda', round=round)

        self.output['mean_curve'].append(np.mean(valid_metrics))
        self.output['var_curve'].append(np.std(valid_metrics))
        # self.output['test_accs'].append(test_metric)
        # self.output['test_losses'].append(test_loss)
        
        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name]=[self.output['valid_accs'][i][cid] for i in range(len(self.output['valid_accs']))]
        
        print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
        print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))
        # print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        # print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
                
        self.max_acc = max(self.max_acc, self.output['mean_curve'][-1])

        # wandb record
        if server.wandb:
            wandb.log(
                {
                    "Mean Client Accuracy": self.output['mean_curve'][-1],
                    "Std Client Accuracy":  self.output['var_curve'][-1],
                    "Max Average Local Accuracy": self.max_acc
                }
            )


logger = MyLogger()

def main():
    multiprocessing.set_start_method('spawn')
    # read options
    option = flw.read_option()
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = '8888'
    os.environ['WORLD_SIZE'] = str(3)
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    
    runname = f"{option['algorithm']}_"
    for para in server.paras_name:
        runname = runname + para + "{}_".format(option[para])
    
    if option['wandb']:
        wandb.init(
            project="HungNN-perFL", 
            entity="aiotlab",
            group=option['task'],
            name=runname[:-1],
            config=option
        )
    
    print("CONFIG =>", option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




