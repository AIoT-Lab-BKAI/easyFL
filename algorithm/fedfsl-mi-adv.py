from msilib.schema import Class
from .fedbase import BasicServer, BasicClient
import torch, copy
from itertools import chain


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def separate_module(model):
    layers = [module for module in model.modules() if not isinstance(module, torch.nn.Sequential)]
    layers = layers[1:]

    feature_generator = torch.nn.Sequential(layers[:-1])
    classifier = layers[-1]

    return feature_generator, classifier

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.feature_generator, self.classifier = separate_module(model)
        return

    def communicate_with(self, client_id):
        svr_pkg = self.pack(client_id)
        if self.clients[client_id].is_drop(): 
            return None
        return self.clients[client_id].reply(svr_pkg)

    def pack(self, client_id):
        return {
            "Fgenerator" : copy.deepcopy(self.feature_generator),
            "Classifier" : copy.deepcopy(self.classifier)
        }

    def unpack(self, packages_received_from_clients):
        Fgenerators = [cp["Fgenerator"] for cp in packages_received_from_clients]
        Classifiers = [cp["Classifier"] for cp in packages_received_from_clients]
        return Fgenerators, Classifiers

    def iterate(self, t):
        self.selected_clients = self.sample()
        Fgenerators, Classifiers = self.communicate(self.selected_clients)

        if not self.selected_clients: 
            return

        impact_factors = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients]
        self.feature_generator = self.aggregate(Fgenerators, p = impact_factors)
        self.classifier = self.aggregate(Classifiers, p = impact_factors)
        return


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.divergence = torch.nn.KLDivLoss()
        self.epochs = int(self.epochs/2)

    def data_to_device(self, data, device=None):
        return data[0].to(device), data[1].to(device)

    def phase_one_training(self, Fgenerator, Classifier, Adv_Classifier, device):
        freeze_model(Fgenerator)
        unfreeze_model(Classifier)
        unfreeze_model(Adv_Classifier)

        Classifier.train(True)
        Adv_Classifier.train(True)
        Fgenerator.train(False)
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = torch.optim.Adam([Classifier.parameters(), Adv_Classifier.parameters()], lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=True)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                Fgenerator.zero_grad()
                Classifier.zero_grad()

                tdata = self.data_to_device(batch_data, device)
                feature = Fgenerator(tdata[0])
                outputs = Classifier(feature.flatten(1))
                adv_outputs = Adv_Classifier(feature.flatten(1))

                loss = self.lossfunc(outputs, tdata[1])
                adv_loss = self.lossfunc(adv_outputs, tdata[1])
                adv_divergence = self.divergence(outputs, adv_outputs)

                total_loss = loss + adv_loss - 0.1 * adv_divergence

                total_loss.backward()
                optimizer.step()
        return

    def phase_two_training(self, Fgenerator, Classifier, Adv_Classifier, device):
        unfreeze_model(Fgenerator)
        freeze_model(Classifier)
        freeze_model(Adv_Classifier)

        Classifier.train(False)
        Adv_Classifier.train(False)
        Fgenerator.train(True)
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = torch.optim.Adam(Fgenerator.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, amsgrad=True)

        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                Fgenerator.zero_grad()
                Classifier.zero_grad()

                tdata = self.data_to_device(batch_data, device)
                feature = Fgenerator(tdata[0])
                outputs = Classifier(feature.flatten(1))
                adv_outputs = Adv_Classifier(feature.flatten(1))

                loss = self.lossfunc(outputs, tdata[1])
                adv_loss = self.lossfunc(adv_outputs, tdata[1])
                adv_divergence = self.divergence(outputs, adv_outputs)

                total_loss = loss + adv_loss + 0.1 * adv_divergence

                total_loss.backward()
                optimizer.step()
        return

    def unpack(self, received_pkg):
        return received_pkg['Fgenerator'], received_pkg['Classifier']

    def reply(self, svr_pkg):
        Fgenerator, Classifier = self.unpack(svr_pkg)
        Adv_Classifier = copy.deepcopy(Classifier)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Fgenerator = Fgenerator.to(device)
        Classifier = Classifier.to(device)
        Adv_Classifier = Adv_Classifier.to(device)

        self.phase_one_training(Fgenerator, Classifier, Adv_Classifier, device)
        self.phase_two_training(Fgenerator, Classifier, Adv_Classifier, device)

        cpkg = self.pack(Fgenerator, Classifier)
        return cpkg

    def pack(self, Fgenerator, Classifier):
        return {
            "Fgenerator" : Fgenerator,
            "Classifier" : Classifier,
        }