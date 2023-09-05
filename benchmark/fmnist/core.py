from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, CusTomTaskReader, DefaultTaskGen, XYTaskReader
import os

class TaskReader(CusTomTaskReader):
    def __init__(self, taskpath='', data_folder="./benchmark/fmnist/data"):
        train_dataset = datasets.FashionMNIST(data_folder, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        test_dataset = datasets.FashionMNIST(data_folder, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        super(TaskReader, self).__init__(os.path.join(taskpath, 'train.json'),train_dataset,test_dataset)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
