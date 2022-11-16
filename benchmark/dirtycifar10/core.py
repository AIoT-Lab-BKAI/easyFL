from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, CusTomTaskReader, DefaultTaskGen, DirtyTaskReader

class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5):
        super(TaskGen, self).__init__(benchmark='cifar10',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/cifar10/data',
                                      )
        self.num_classes = 10
        self.save_data = self.XYData_to_json

    def load_data(self):
        self.train_data = datasets.MNIST(self.rawdata_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        self.test_data = datasets.MNIST(self.rawdata_path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))

    def convert_data_for_saving(self):
        train_x = [self.train_data[did][0].tolist() for did in range(len(self.train_data))]
        train_y = [self.train_data[did][1] for did in range(len(self.train_data))]
        test_x = [self.test_data[did][0].tolist() for did in range(len(self.test_data))]
        test_y = [self.test_data[did][1] for did in range(len(self.test_data))]
        self.train_data = {'x':train_x, 'y':train_y}
        self.test_data = {'x': test_x, 'y': test_y}
        return


class TaskReader(DirtyTaskReader):
    def __init__(self, taskpath, train_dataset=None, test_dataset=None, noise_magnitude=1, dirty_rate=None, data_folder="./benchmark/cifar10/data", noise_type='', option=None):
        train_dataset = datasets.CIFAR10(data_folder, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_dataset = datasets.CIFAR10(data_folder, train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        super().__init__(taskpath, train_dataset, test_dataset, noise_magnitude, dirty_rate, noise_type, option)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)

