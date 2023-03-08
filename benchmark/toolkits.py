"""
DISTRIBUTION OF DATASET
-----------------------------------------------------------------------------------
balance:
    iid:            0 : identical and independent distributions of the dataset among clients
    label skew:     1 Quantity:  each party owns data samples of a fixed number of labels.
                    2 Dirichlet: each party is allocated a proportion of the samples of each label according to Dirichlet distribution.
                    3 Shard: each party is allocated the same numbers of shards that is sorted by the labels of the data
-----------------------------------------------------------------------------------
depends on partitions:
    feature skew:   4 Noise: each party owns data samples of a fixed number of labels.
                    5 ID: For Shakespeare\FEMNIST, we divide and assign the writers (and their characters) into each party randomly and equally.
-----------------------------------------------------------------------------------
imbalance:
    iid:            6 Vol: only the vol of local dataset varies.
    niid:           7 Vol: for generating synthetic data
"""
from builtins import breakpoint
from dataclasses import replace
from pathlib import Path
import torch
import ujson
import numpy as np
import os.path
import random
import urllib
import zipfile
import os
import ssl
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
ssl._create_default_https_context = ssl._create_unverified_context
import importlib
from torchvision import transforms, datasets
import json
import os
from PIL import Image
import time
from benchmark.uncertainty_loss import one_hot_embedding, edl_mse_loss
import csv

def set_random_seed(seed=0):
    """Set random seed"""
    random.seed(3 + seed)
    np.random.seed(97 + seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url: urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

class BasicTaskGen:
    _TYPE_DIST = {
        0: 'iid',
        1: 'label_skew_quantity',
        2: 'label_skew_dirichlet',
        3: 'label_skew_shard',
        4: 'feature_skew_noise',
        5: 'feature_skew_id',
        6: 'iid_volumn_skew',
        7: 'niid_volumn_skew',
        8: 'concept skew',
        9: 'concept and feature skew and balance',
        10: 'concept and feature skew and imbalance',
    }
    _TYPE_DATASET = ['2DImage', '3DImage', 'Text', 'Sequential', 'Graph', 'Tabular']

    def __init__(self, benchmark, dist_id, skewness, rawdata_path, seed=0):
        self.benchmark = benchmark
        self.rootpath = './fedtask'
        self.rawdata_path = rawdata_path
        self.dist_id = dist_id
        self.dist_name = self._TYPE_DIST[dist_id]
        self.skewness = 0 if dist_id==0 else skewness
        self.num_clients = -1
        self.seed = seed
        set_random_seed(self.seed)

    def run(self):
        """The whole process to generate federated task. """
        pass

    def load_data(self):
        """Download and load dataset into memory."""
        pass

    def partition(self):
        """Partition the data according to 'dist' and 'skewness'"""
        pass

    def save_data(self):
        """Save the federated dataset to the task_path/data.
        This algorithm should be implemented as the way to read
        data from disk that is defined by DataReader.read_data()
        """
        pass

    def save_info(self):
        """Save the task infomation to the .json file stored in taskpath"""
        pass

    def get_taskname(self):
        """Create task name and return it."""
        taskname = '_'.join([self.benchmark, 'cnum' +  str(self.num_clients), 'dist' + str(self.dist_id), 'skew' + str(self.skewness).replace(" ", ""), 'seed'+str(self.seed)])
        return taskname

    def get_client_names(self):
        k = str(len(str(self.num_clients)))
        return [('Client{:0>' + k + 'd}').format(i) for i in range(self.num_clients)]

    def create_task_directories(self):
        """Create the directories of the task."""
        taskname = self.get_taskname()
        taskpath = os.path.join(self.rootpath, taskname)
        os.mkdir(taskpath)
        os.mkdir(os.path.join(taskpath, 'record'))

    def _check_task_exist(self):
        """Check whether the task already exists."""
        taskname = self.get_taskname()
        return os.path.exists(os.path.join(self.rootpath, taskname))

class DefaultTaskGen(BasicTaskGen):
    def __init__(self, benchmark, dist_id, skewness, rawdata_path, num_clients=1, minvol=10, seed=0):
        super(DefaultTaskGen, self).__init__(benchmark, dist_id, skewness, rawdata_path, seed)
        self.minvol=minvol
        self.num_classes = -1
        self.train_data = None
        self.test_data = None
        self.num_clients = num_clients
        self.cnames = self.get_client_names()
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.rootpath, self.taskname)
        self.save_data = self.XYData_to_json
        self.datasrc = {
            'lib': None,
            'class_name': None,
            'args':[]
        }

    def run(self):
        """ Generate federated task"""
        # check if the task exists
        if not self._check_task_exist():
            self.create_task_directories()
        else:
            print("Task Already Exists.")
            return
        # read raw_data into self.train_data and self.test_data
        print('-----------------------------------------------------')
        print('Loading...')
        self.load_data()
        print('Done.')
        # partition data and hold-out for each local dataset
        print('-----------------------------------------------------')
        print('Partitioning data...')
        local_datas = self.partition()
        train_cidxs, valid_cidxs = self.local_holdout(local_datas, rate=0.8, shuffle=True)
        print('Done.')
        # save task infomation as .json file and the federated dataset
        print('-----------------------------------------------------')
        print('Saving data...')
        self.save_info()
        self.save_data(train_cidxs, valid_cidxs)
        print('Done.')
        return

    def load_data(self):
        """ load and pre-process the raw data"""
        return

    def partition(self):
        # Partition self.train_data according to the delimiter and return indexes of data owned by each client as [c1data_idxs, ...] where the type of each element is list(int)
        if self.dist_id == 0:
            """IID"""
            d_idxs = np.random.permutation(len(self.train_data))
            local_datas = np.array_split(d_idxs, self.num_clients)

        elif self.dist_id == 1:
            """label_skew_quantity"""
            self.skewness = min(max(0, self.skewness),1.0)
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            num = max(int((1-self.skewness)*self.num_classes), 1)
            K = self.num_classes
            local_datas = [[] for _ in range(self.num_clients)]
            if num == K:
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, self.num_clients)
                    for cid in range(self.num_clients):
                        local_datas[cid].extend(split[cid].tolist())
            else:
                times = [0 for _ in range(self.num_classes)]
                contain = []
                for i in range(self.num_clients):
                    current = [i % K]
                    times[i % K] += 1
                    j = 1
                    while (j < num):
                        ind = random.randint(0, K - 1)
                        if (ind not in current):
                            j = j + 1
                            current.append(ind)
                            times[ind] += 1
                    contain.append(current)
                for k in range(K):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    split = np.array_split(idx_k, times[k])
                    ids = 0
                    for cid in range(self.num_clients):
                        if k in contain[cid]:
                            local_datas[cid].extend(split[ids].tolist())
                            ids += 1

        elif self.dist_id == 2:
            """label_skew_dirichlet"""
            min_size = 0
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            local_datas = [[] for _ in range(self.num_clients)]
            while min_size < self.minvol:
                idx_batch = [[] for i in range(self.num_clients)]
                for k in range(self.num_classes):
                    idx_k = [p[0] for p in dpairs if p[1]==k]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                    ## Balance
                    proportions = np.array([p * (len(idx_j) < len(self.train_data)/ self.num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
            for j in range(self.num_clients):
                np.random.shuffle(idx_batch[j])
                local_datas[j].extend(idx_batch[j])

        elif self.dist_id == 3:
            """label_skew_shard"""
            dpairs = [[did, self.train_data[did][-1]] for did in range(len(self.train_data))]
            self.skewness = min(max(0, self.skewness), 1.0)
            num_shards = max(int((1 - self.skewness) * self.num_classes * 2), 1)
            client_datasize = int(len(self.train_data) / self.num_clients)
            all_idxs = [i for i in range(len(self.train_data))]
            z = zip([p[1] for p in dpairs], all_idxs)
            z = sorted(z)
            labels, all_idxs = zip(*z)
            shardsize = int(client_datasize / num_shards)
            idxs_shard = range(int(self.num_clients * num_shards))
            local_datas = [[] for i in range(self.num_clients)]
            for i in range(self.num_clients):
                rand_set = set(np.random.choice(idxs_shard, num_shards, replace=False))
                idxs_shard = list(set(idxs_shard) - rand_set)
                for rand in rand_set:
                    local_datas[i].extend(all_idxs[rand * shardsize:(rand + 1) * shardsize])

        elif self.dist_id == 4:
            pass

        elif self.dist_id == 5:
            """feature_skew_id"""
            if not isinstance(self.train_data, TupleDataset):
                raise RuntimeError("Support for dist_id=5 only after setting the type of self.train_data is TupleDataset")
            Xs, IDs, Ys = self.train_data.tolist()
            self.num_clients = len(set(IDs))
            local_datas = [[] for _ in range(self.num_clients)]
            for did in range(len(IDs)):
                local_datas[IDs[did]].append(did)

        elif self.dist_id == 6:
            minv = 0
            d_idxs = np.random.permutation(len(self.train_data))
            while minv < self.minvol:
                proportions = np.random.dirichlet(np.repeat(self.skewness, self.num_clients))
                proportions = proportions / proportions.sum()
                minv = np.min(proportions * len(self.train_data))
            proportions = (np.cumsum(proportions) * len(d_idxs)).astype(int)[:-1]
            local_datas  = np.split(d_idxs, proportions)
        return local_datas

    def local_holdout(self, local_datas, rate=0.8, shuffle=False):
        """split each local dataset into train data and valid data according the rate."""
        train_cidxs = []
        valid_cidxs = []
        for local_data in local_datas:
            if shuffle:
                np.random.shuffle(local_data)
            k = int(len(local_data) * rate)
            train_cidxs.append(local_data[:k])
            valid_cidxs.append(local_data[k:])
        return train_cidxs, valid_cidxs

    def save_info(self):
        info = {
            'benchmark': self.benchmark,  # name of the dataset
            'dist': self.dist_id,  # type of the partition way
            'skewness': self.skewness,  # hyper-parameter for controlling the degree of niid
            'num-clients': self.num_clients,  # numbers of all the clients
        }
        # save info.json
        with open(os.path.join(self.taskpath, 'info.json'), 'w') as outf:
            ujson.dump(info, outf)

    def convert_data_for_saving(self):
        """Convert self.train_data and self.test_data to list that can be stored as .json file and the converted dataset={'x':[], 'y':[]}"""
        pass

    def XYData_to_json(self, train_cidxs, valid_cidxs):
        self.convert_data_for_saving()
        # save federated dataset
        feddata = {
            'store': 'XY',
            'client_names': self.cnames,
            'dtest': self.test_data

        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain':{
                    'x':[self.train_data['x'][did] for did in train_cidxs[cid]], 'y':[self.train_data['y'][did] for did in train_cidxs[cid]]
                },
                'dvalid':{
                    'x':[self.train_data['x'][did] for did in valid_cidxs[cid]], 'y':[self.train_data['y'][did] for did in valid_cidxs[cid]]
                }
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    def IDXData_to_json(self, train_cidxs, valid_cidxs):
        if self.datasrc ==None:
            raise RuntimeError("Attr datasrc not Found. Please define it in __init__() before calling IndexData_to_json")
        feddata = {
            'store': 'IDX',
            'client_names': self.cnames,
            'dtest': [i for i in range(len(self.test_data))],
            'datasrc': self.datasrc
        }
        for cid in range(self.num_clients):
            feddata[self.cnames[cid]] = {
                'dtrain': train_cidxs[cid],
                'dvalid': valid_cidxs[cid]
            }
        with open(os.path.join(self.taskpath, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

class BasicTaskCalculator:

    _OPTIM = None

    def __init__(self, device):
        self.device = device
        self.lossfunc = None
        self.DataLoader = None

    def data_to_device(self, data):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_evaluation(self):
        raise NotImplementedError

    def get_data_loader(self, data, batch_size = 64):
        return NotImplementedError

    def test(self):
        raise NotImplementedError

    def get_optimizer(self, name="sgd", model=None, lr=0.1, weight_decay=0, momentum=0):
        # if self._OPTIM == None:
        #     raise RuntimeError("TaskCalculator._OPTIM Not Initialized.")
        if name.lower() == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif name.lower() == 'adam':
            return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise RuntimeError("Invalid Optimizer.")

    @classmethod
    def setOP(cls, OP):
        cls._OPTIM = OP

class ClassifyCalculator(BasicTaskCalculator):
    def __init__(self, device):
        super(ClassifyCalculator, self).__init__(device)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.num_classes = 10
        self.DataLoader = DataLoader
        self.lossfunc_Dir = DirichletLoss(num_classes= self.num_classes, annealing_step= 10, device= device)


    def get_loss_not_uncertainty(self, model, data, device=None):
        tdata = self.data_to_device(data, device)
        outputs = model(tdata[0])
        softmax = torch.nn.Softmax(dim=1)
        # print('outputs =', outputs)
        # print('torch.max(softmax(outputs), dim=1)[0] =', torch.max(softmax(outputs), dim=1)[0])
        # print('tdata[2] =', tdata[2])
        # breakpoint()
        # for i in torch.max(softmax(outputs), dim=1)[0].tolist():
        #     if i > 1.0:
        #         print('output :', outputs)
        #         print('output :', max(softmax(outputs)))
        #         print('output :', torch.max(softmax(outputs), dim=1)[0])
        #         break
        loss = self.lossfunc(outputs, tdata[1])
        return loss#, torch.max(softmax(outputs), dim=1)[0].tolist(), tdata[2].tolist()
    
    def get_loss(self, model, data, epoch, device=None):
        tdata = self.data_to_device(data, device)
        outputs = model(tdata[0]) # batchsize * num_class
        y = one_hot_embedding(tdata[1], self.num_classes)
        # loss = self.lossfunc(outputs, tdata[1])
        # loss = self.lossMSE(outputs, tdata[1].float())
        loss = self.lossfunc_Dir(outputs, y.float(), epoch)
        ## compute uncertainty
        evidence = F.relu(outputs)
        alpha = evidence + 1
        # print(alpha)
        u = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        unc = torch.sum(u)
        return loss, unc

    def get_uncertainty(self, model, data):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = data.to(device)
        model = model.to(device)
        output = model(torch.unsqueeze(data, dim=0))
        evidence = F.relu(output)
        alpha = evidence + 1
        uncertainty = self.num_classes/ torch.sum(alpha, dim=1)
        return uncertainty, torch.sum(output, dim = 1)

    @torch.no_grad()
    def get_evaluation(self, model, data):
        tdata = self.data_to_device(data)
        outputs = model(tdata)
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item()

    @torch.no_grad()
    def test(self, model, data, device=None):
        """Metric = Accuracy"""
        tdata = self.data_to_device_test(data, device)
        model = model.to(device)
        start = time.time()
        outputs = model(tdata[0])
        end = time.time()
        loss = self.lossfunc(outputs, tdata[-1])
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item(), (end - start)

    @torch.no_grad()
    def test_client(self, model, data, device=None):
        """Metric = Accuracy"""
        tdata = self.data_to_device(data, device)
        model = model.to(device)
        softmax = torch.nn.Softmax(dim=1)
        start = time.time()
        outputs = model(tdata[0])
        end = time.time()
        loss = self.lossfunc(outputs, tdata[1])
        y_pred = outputs.data.max(1, keepdim=True)[1]
        correct = y_pred.eq(tdata[1].data.view_as(y_pred)).long().cpu().sum()
        return (1.0 * correct / len(tdata[1])).item(), loss.item(), (end - start), torch.max(softmax(outputs), dim=1)[0].tolist(), tdata[2].tolist()

    def data_to_device(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device), data[2]
        else:
            return data[0].to(device), data[1].to(device), data[2]

    def data_to_device_test(self, data, device=None):
        if device is None:
            return data[0].to(self.device), data[1].to(self.device)
        else:
            return data[0].to(device), data[1].to(device)

    def get_data_loader(self, dataset, batch_size=64, shuffle=True, droplast=False):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=droplast)

class DirichletLoss(nn.Module):
    def __init__(self, num_classes, annealing_step, device):
        super(DirichletLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        self.device = device
        
    def forward(self, output, target, epoch_num):
        return edl_mse_loss(output, target, epoch_num, self.num_classes, self.annealing_step, self.device)

class BasicTaskReader:
    def __init__(self, taskpath=''):
        self.taskpath = taskpath

    def read_data(self):
        """
            Reading the spilted dataset from disk files and loading data into the class 'LocalDataset'.
            This algorithm should read three types of data from the processed task:
                train_sets = [client1_train_data, ...] where each item is an instance of 'LocalDataset'
                valid_sets = [client1_valid_data, ...] where each item is an instance of 'LocalDataset'
                test_set = test_dataset
            Return train_sets, valid_sets, test_set, client_names
        """
        pass

class XYTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(XYTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        test_data = XYDataset(feddata['dtest']['x'], feddata['dtest']['y'])
        train_datas = [XYDataset(feddata[name]['dtrain']['x'], feddata[name]['dtrain']['y']) for name in feddata['client_names']]
        valid_datas = [XYDataset(feddata[name]['dvalid']['x'], feddata[name]['dvalid']['y']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']
    
class CusTomTaskReader(BasicTaskReader):
    def __init__(self, taskpath, train_dataset, test_dataset):
        super().__init__(taskpath)
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.taskpath = taskpath
    
    def load_dataset_idx(self,path="data"):
        import json
        list_idx = json.load(open(path, 'r'))
        return {int(k): v for k, v in list_idx.items()}
    
    def read_data(self):
        data_idx = self.load_dataset_idx('dataset_idx/'+ self.taskpath)
        n_clients = len(data_idx)
        train_data = [CustomDataset(self.train_dataset,data_idx[idx]) for idx in range(n_clients)]
        test_data = self.test_dataset
        return train_data, test_data, n_clients


class DirtyTaskReader(BasicTaskReader):        
    def __init__(self, taskpath, train_dataset, test_dataset, noise_magnitude=1, dirty_rate=None, noise_type='', option=None):
        super().__init__(taskpath)
        self.noise_magnitude = noise_magnitude
        self.dirty_rate = dirty_rate
        self.test_dataset = test_dataset
        self.train_dataset = train_dataset
        self.taskpath = taskpath
        self.noise_type = noise_type
        self.option = option
    
    def load_dataset_idx(self,path="data"):
        import json
        list_idx = json.load(open(path, 'r'))
        return {int(k): v for k, v in list_idx.items()}
        
    def read_data(self):
        data_idx = self.load_dataset_idx('dataset_idx/'+ self.taskpath)
        n_clients = len(data_idx)
        train_datas = [DirtyDataset(self.train_dataset, 
                                    data_idx[idx], 
                                    seed=idx, 
                                    magnitude=self.noise_magnitude,
                                    dirty_rate=self.dirty_rate[idx], noise_type=self.noise_type, option=self.option) for idx in range(n_clients)]
        test_data = self.test_dataset
        print("Here return dirty training datasets for clients, clean test dataset for server")
        return train_datas, test_data, n_clients
    
    
class IDXTaskReader(BasicTaskReader):
    def __init__(self, taskpath=''):
        super(IDXTaskReader, self).__init__(taskpath)

    def read_data(self):
        with open(os.path.join(self.taskpath, 'data.json'), 'r') as inf:
            feddata = ujson.load(inf)
        DS = getattr(importlib.import_module(feddata['datasrc']['lib']), feddata['datasrc']['class_name'])
        arg_strings = '(' + ','.join(feddata['datasrc']['args'])
        train_args = arg_strings + ', train=True)'
        test_args = arg_strings + ', train=False)'
        DS.SET_DATA(eval(feddata['datasrc']['class_name'] + train_args))
        DS.SET_DATA(eval(feddata['datasrc']['class_name'] + test_args), key='TEST')
        test_data = IDXDataset(feddata['dtest'], key='TEST')
        train_datas = [IDXDataset(feddata[name]['dtrain']) for name in feddata['client_names']]
        valid_datas = [IDXDataset(feddata[name]['dvalid']) for name in feddata['client_names']]
        return train_datas, valid_datas, test_data, feddata['client_names']


class PillDataset(Dataset):
    def __init__(self,user_idx,img_folder_path="",idx_dict=None,label_dict=None,map_label_dict=None):
        super().__init__()
        self.user_idx = user_idx
        self.idx = idx_dict[str(user_idx)]
        self.img_folder_path = img_folder_path
        self.label_dict = label_dict
        self.map_label_dict = map_label_dict
        self.transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])

    def __len__(self):
        # return 1
        return len(self.idx)

    def __getitem__(self, item):
        img_name = self.idx[item]
        img_path = os.path.join(self.img_folder_path,img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        pill_name = self.label_dict[img_name]
        label = self.map_label_dict[pill_name]
        return img,label 


class CustomDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


import matplotlib.pyplot as plt
import torchvision.transforms as T

def imshow(img, dir = "{self.option['log_folder']}/pics", name="img.png"):
    if not Path(name).exists():
        os.system(f"mkdir -p {dir}")
    # breakpoint()
    # plt.imshow(img)
    plt.imshow(img.permute(1,2,0))
    plt.savefig(dir + "/" + name)

class DirtyDataset(Dataset):
    count = 0
    def __init__(self, dataset, idxs, seed, dirty_rate=0.2, magnitude=1, noise_type='', option=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.origin_idxs = list(idxs)
        self.seed = seed
        dirty_quantity = int(dirty_rate * len(self.idxs))
        self.magnitude = magnitude
        self.noise_type = noise_type
        self.option = option
        self.dirty_rate = dirty_rate
        if self.dirty_rate != 0:
            self.client_type = 'attacker'
        else:
            self.client_type = 'benign'
        
        
        np.random.seed(self.seed)
        self.dirty_dataidx = np.random.choice(self.idxs, dirty_quantity, replace=False).tolist()
        self.clean_dataidx = list(set(self.idxs) - set(self.dirty_dataidx))
        self.type_image_idx = {
            'noise': self.dirty_dataidx,
            'clean': self.clean_dataidx
        }
        # if seed in [0, 1]:
        #     print(f'client {seed}, dirty {self.dirty_dataidx}')
        # path_dirty_dataidx = f'./results/dirty_dataidx/10000data_dirty_rate_{dirty_rate}'
        # if not os.path.exists(path_dirty_dataidx):
        #     os.makedirs(path_dirty_dataidx)
        # with open(path_dirty_dataidx + '/' + f'{seed}.json', 'w') as f:
        #     json.dump(self.dirty_dataidx, f)
        
        # with open(f'predict_unclean_idx/{seed}.json', 'r') as f:
        #     predict_unclean_idx = json.load(f)
        # np.random.seed(self.seed)
        # self.idxs = list(set(self.idxs) - set(np.random.choice(self.dirty_dataidx, int(0*len(self.dirty_dataidx)), replace=False).tolist()))
        self.rotater = T.RandomRotation(degrees=(90, 270))
        # self.resize_cropper = T.RandomResizedCrop(size=sample.shape)
        self.blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(9, 11))
        self.addgaussiannoise = AddGaussianNoise(magnitude, 0.5, self.seed)
        from torchvision.transforms import Grayscale
        from skimage.util import random_noise
        self.gray_transform = Grayscale(num_output_channels=3)
        # self.beta = 1
        # self.beta_idxs = self.idxs
        # if self.seed == 5:
        #     self.unintersection = [799,10367,24040,4525,9740,30433,34981,38428,49085,38691,15420,19526,31483,39870,6922,1136,22780,16351,18410,4409]
            # with open('./results/csv/unintersection.csv', 'r') as f:
            #     # reader = csv.reader(f)
            #     bre
            #     for row in reader:
                    # self.unintersection= row

    def remove_noise(self, percent):
        self.idxs = list(set(self.idxs) - set(np.random.choice(self.dirty_dataidx, int(percent*len(self.dirty_dataidx)), replace=False).tolist()))
    
    def remove_noise_specific(self, list_noise):
        self.idxs = list(set(self.idxs) - set(list_noise))

    def reverse_idx(self):
        self.idxs = self.origin_idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        # random.seed(self.seed) # apply this seed to img transforms
        image, label = self.dataset[self.idxs[item]]
        if self.client_type == "attacker":
            if label in self.option['attacked_class']:
                # torch.manual_seed(self.idxs[item])
                # rand = torch.randn(1)
                # if rand <= self.dirty_rate:
                if self.idxs[item] in self.dirty_dataidx:
                    # self.type_image_idx['noise'].add(self.idxs[item])
                    if self.noise_type == 'gaussian':
                        if DirtyDataset.count < 2:
                            imshow(image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_before.png")
                        torch.manual_seed(self.idxs[item])
                        noise_image = torch.randn(image.size()) + self.magnitude
                        # noise_image = self.blurrer(self.addgaussiannoise(self.rotater(noise_image)))
                        if self.option['outside_noise'] == 'inside':
                            noise_image += image
                        noise_image = (noise_image - torch.min(noise_image))/(torch.max(noise_image) - torch.min(noise_image))
                        if DirtyDataset.count < 2:
                            imshow(noise_image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_after.png")
                            DirtyDataset.count += 1
                        return noise_image, label, self.idxs[item]
                    
                    #salt&peppernoise
                    elif self.noise_type == 'salt_pepper':
                        if DirtyDataset.count < 2:
                            imshow(image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_before.png")
                        torch.manual_seed(self.idxs[item])
                        noisy_mask = torch.randint(low=0, high=int(1/self.magnitude)+1, size=image.size())
                
                        zeros_pixel = np.where(noisy_mask == 0)
                        one_pixel = np.where(noisy_mask == int(1/self.magnitude))
                        noise_image = self.gray_transform(image)
                        # print(torch.max(image))
                        # print(torch.min(image))
                        noise_image[zeros_pixel] = 0.0
                        noise_image[one_pixel] = 1.0
                    
                        if DirtyDataset.count < 2:
                            imshow(noise_image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_after.png")
                            DirtyDataset.count += 1
                        return noise_image, label, self.idxs[item]
                    
                    #speckle noise
                    elif self.noise_type == 'speckle':
                        if DirtyDataset.count < 2:
                            imshow(image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_before.png")
                        torch.manual_seed(self.idxs[item])
                        noisy_mask = torch.randn(image.size()) + self.magnitude
                        
                        noise_image = image * noisy_mask
                        noise_image = (noise_image - torch.min(noise_image))/(torch.max(noise_image) - torch.min(noise_image))
                        if DirtyDataset.count < 2:
                            imshow(noise_image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_after.png")
                            DirtyDataset.count += 1
                        return noise_image, label, self.idxs[item]
                    
                    # poisson noise
                    elif self.noise_type == 'poisson':
                        if DirtyDataset.count < 2:
                            imshow(image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_before.png")
                        torch.manual_seed(self.idxs[item])
                        noise_mask = torch.poisson(torch.rand(image.size())*self.magnitude)
                        noise_image = image + noise_mask
                        noise_image = (noise_image - torch.min(noise_image))/(torch.max(noise_image) - torch.min(noise_image))
                        if DirtyDataset.count < 2:
                            imshow(noise_image, f"{self.option['log_folder']}/pics/{self.noise_type}", f"{self.idxs[item]}_after.png")
                            DirtyDataset.count += 1
                        return noise_image, label, self.idxs[item]
                    
        #         else:
        #             self.type_image_idx['clean'].add(self.idxs[item])
        #     else:
        #         self.type_image_idx['clean'].add(self.idxs[item])
        # else:
        #     self.type_image_idx['clean'].add(self.idxs[item])
        return image, label, self.idxs[item]
        # if self.idxs[item] in self.dirty_dataidx:
        #     if DirtyDataset.count < 2:
        #         imshow(image, f"{self.option['log_folder']}/pics/noise_{self.noise_type}", f"{self.idxs[item]}_before.png")
        #     # # image = image + self.noise
            
        #     #gaussian noise
        #     if self.noise_type == 'gaussian':
        #         torch.manual_seed(item)
        #         noise_image = torch.randn(image.size())
        #         noise_image = self.blurrer(self.addgaussiannoise(self.rotater(noise_image)))
        #         noise_image = (noise_image - torch.min(noise_image))/(torch.max(noise_image) - torch.min(noise_image))
        #         # noise_image = torch.clamp(noise_image, min=0, max=1)

        #     #salt&peppernoise
        #     elif self.noise_type == 'salt_pepper':
        #         torch.manual_seed(item)
        #         noisy_mask = torch.randint(low=0, high=int(1/self.magnitude)+1, size=image.size())
                
        #         zeros_pixel = np.where(noisy_mask == 0)
        #         one_pixel = np.where(noisy_mask == int(1/self.magnitude))
        #         noise_image = self.gray_transform(image)
        #         # print(torch.max(image))
        #         # print(torch.min(image))
        #         noise_image[zeros_pixel] = 0
        #         noise_image[one_pixel] = 1
            
        #     #speckle noise
        #     elif self.noise_type == 'speckle':
        #         torch.manual_seed(item)
        #         noisy_mask = torch.randn(image.size()) + self.magnitude
                
        #         noise_image = image * noisy_mask
        #         noise_image = (noise_image - torch.min(noise_image))/(torch.max(noise_image) - torch.min(noise_image))
            
        #     # poisson noise
        #     elif self.noise_type == 'poisson':
        #         torch.manual_seed(item)
        #         noise_mask = torch.poisson(torch.rand(image.size())*self.magnitude)
        #         noise_image = image + noise_mask
        #         noise_image = (noise_image - torch.min(noise_image))/(torch.max(noise_image) - torch.min(noise_image))
            
        #     # mix
        #     else:
        #         pass
        #     # self.dirty_dataidx.remove(self.idxs[item])
        #     if DirtyDataset.count < 2:
        #         imshow(noise_image, f"{self.option['log_folder']}/pics/noise_{self.noise_type}", f"{self.idxs[item]}_after.png")
        #         DirtyDataset.count += 1
            
        #     return noise_image, label
        
        # return image, label

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., seed=1000):
        self.std = std
        self.mean = mean
        self.seed = seed
        
    def __call__(self, tensor):
        torch.manual_seed(self.seed)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class XYDataset(Dataset):
    def __init__(self, X=[], Y=[], totensor = True):
        """ Init Dataset with pairs of features and labels/annotations.
        XYDataset transforms data that is list\array into tensor.
        The data is already loaded into memory before passing into XYDataset.__init__()
        and thus is only suitable for benchmarks with small size (e.g. CIFAR10, MNIST)
        Args:
            X: a list of features
            Y: a list of labels with the same length of X
        """
        if not self._check_equal_length(X, Y):
            raise RuntimeError("Different length of Y with X.")
        if totensor:
            try:
                self.X = torch.tensor(X)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X = X
            self.Y = Y
        self.all_labels = list(set(self.tolist()[1]))

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

    def tolist(self):
        if not isinstance(self.X, torch.Tensor):
            return self.X, self.Y
        return self.X.tolist(), self.Y.tolist()

    def _check_equal_length(self, X, Y):
        return len(X)==len(Y)

    def get_all_labels(self):
        return self.all_labels

class IDXDataset(Dataset):
    # The source dataset that can be indexed by IDXDataset
    _DATA = {'TRAIN': None,'TEST': None}

    def __init__(self, idxs, key='TRAIN'):
        """Init dataset with 'src_data' and a list of indexes that are used to position data in 'src_data'"""
        if not isinstance(idxs, list):
            raise RuntimeError("Invalid Indexes")
        self.idxs = idxs
        self.key = key

    @classmethod
    def SET_DATA(cls, dataset, key = 'TRAIN'):
        cls._DATA[key] = dataset

    @classmethod
    def ADD_KEY_TO_DATA(cls, key, value = None):
        if key==None:
            raise RuntimeError("Empty key when calling class algorithm IDXData.ADD_KEY_TO_DATA")
        cls._DATA[key]=value

    def __getitem__(self, item):
        idx = self.idxs[item]
        return self._DATA[self.key][idx]

class TupleDataset(Dataset):
    def __init__(self, X1=[], X2=[], Y=[], totensor=True):
        if totensor:
            try:
                self.X1 = torch.tensor(X1)
                self.X2 = torch.tensor(X2)
                self.Y = torch.tensor(Y)
            except:
                raise RuntimeError("Failed to convert input into torch.Tensor.")
        else:
            self.X1 = X1
            self.X2 = X2
            self.Y = Y

    def __getitem__(self, item):
        return self.X1[item], self.X2[item], self.Y[item]

    def __len__(self):
        return len(self.Y)

    def tolist(self):
        if not isinstance(self.X1, torch.Tensor):
            return self.X1, self.X2, self.Y
        return self.X1.tolist(), self.X2.tolist(), self.Y.tolist()
