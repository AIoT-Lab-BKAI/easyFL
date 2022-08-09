from torchvision import datasets, transforms, ImageFolder
from benchmark.toolkits import (
    ClassifyCalculator,
    CusTomTaskReader,
    DefaultTaskGen,
    DirtyTaskReader,
)


class TaskReader(DirtyTaskReader):
    def __init__(
        self,
        taskpath,
        train_dataset=None,
        test_dataset=None,
        noise_magnitude=1,
        dirty_rate=None,
        data_folder="./benchmark/mnist/data",
    ):
        # train_dataset = datasets.MNIST(data_folder, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        # test_dataset = datasets.MNIST(data_folder, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]))
        train_dataset = ImageFolder(
            "data/subdataset/train",
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            ),
        )
        test_dataset = ImageFolder(
            "data/subdataset/test",
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize((224, 224))]
            ),
        )
        super().__init__(
            taskpath, train_dataset, test_dataset, noise_magnitude, dirty_rate
        )
