import math
from torch import nn
from utils.fmodule import FModule
import torch.nn.functional as F

cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512,"M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
    }

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.features = make_layers(cfg["A"])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 100),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                m.bias.data.zero_()    
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def pred_and_pre(self, x):
        x = self.features(x)
        e = x.view(x.size(0), -1)
        o = self.classifier(e)
        return o, e


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)



# def vgg11_mnist(output_dim):
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers_mnist(cfg["A"],output_dim))

# def vgg11(output_dim):
#     """VGG 11-layer model (configuration "A")"""
#     return VGG(make_layers(cfg["A"],output_dim))

# def cifar_vgg19():
#     """VGG 19-layer model (configuration 'E')"""
#     return VGG(make_layers(cfg["E"]))

# def vgg11_bn():
#     """VGG 11-layer model (configuration "A") with batch normalization"""
#     return VGG(make_layers(cfg["A"], batch_norm=True))


# def vgg13():
#     """VGG 13-layer model (configuration "B")"""
#     return VGG(make_layers(cfg["B"]))


# def vgg13_bn():
#     """VGG 13-layer model (configuration "B") with batch normalization"""
#     return VGG(make_layers(cfg["B"], batch_norm=True))


# def vgg16():
#     """VGG 16-layer model (configuration "D")"""
#     return VGG(make_layers(cfg["D"]))


# def vgg16_bn():
#     """VGG 16-layer model (configuration "D") with batch normalization"""
#     return VGG(make_layers(cfg["D"], batch_norm=True))


# def vgg19():
#     """VGG 19-layer model (configuration "E")"""
#     return VGG(make_layers(cfg["E"]))


# def vgg19_bn():
#     """VGG 19-layer model (configuration 'E') with batch normalization"""
#     return VGG(make_layers(cfg["E"], batch_norm=True))