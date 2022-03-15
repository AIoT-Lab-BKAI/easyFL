from algorithm.fedbase import BasicServer, BasicClient
import torch.nn as nn


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = nn.CrossEntropyLoss()
        self.train_data_list = self.div_train_data()
        
        
    def div_train_data(self):
        pass


    def train(self, model):
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                loss = self.get_loss(model, batch_data)
                loss.backward()
                optimizer.step()
        return


    def get_loss(self, model, data):
        separate_data = []
        for x, y in zip(data.X, data.Y):
            pass
        tdata = self.data_to_device(data)
        outputs = model(tdata[0])
        loss = self.lossfunc(outputs, tdata[1])
        return loss