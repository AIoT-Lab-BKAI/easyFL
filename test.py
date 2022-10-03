import argparse
from ast import arg
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from benchmark.mnist.model.cnn import Model
import os
from torchmetrics import ConfusionMatrix
import numpy as np

testing_data = datasets.MNIST(
    root="./benchmark/mnist/data",
    train=False,
    download=False,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
)

def acc_test(model, testing_data):
    test_loader = DataLoader(testing_data, batch_size=32, shuffle=True, drop_last=False)
    model.cuda()
    model.eval()

    loss_fn = torch.nn.CrossEntropyLoss()
    device = "cuda"

    size = len(test_loader.dataset)
    num_batches = len(test_loader)
    model.eval()
    test_loss, correct = 0, 0
    confmat = ConfusionMatrix(num_classes=10).to(device)
    cmtx = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            cmtx += confmat(pred, y)

    test_loss /= num_batches
    correct /= size
    return 100 * correct, test_loss, cmtx.cpu().numpy()


def folder_test(folder_path):
    for f in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, f)):
            folder_test(os.path.join(folder_path,f))
        else:
            if ".pt" not in f:
                continue

            print("Test", os.path.join(folder_path,f))
            model = Model()
            model.load_state_dict(torch.load(os.path.join(folder_path,f)))
            acc, loss, cfmtx = acc_test(model, testing_data)
            cfmtx = cfmtx/np.sum(cfmtx, axis=1, keepdims=True)
            np.savetxt(os.path.join(folder_path, f.split('.')[0] + "_cfmtx.txt"), cfmtx, fmt='%.2f', delimiter=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="models")
    args = parser.parse_args()

    folder_test(args.folder)
