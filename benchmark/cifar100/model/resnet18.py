import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fmodule import FModule
from torchvision.models import resnet18, ResNet18_Weights

# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels)  # Calculate loss
#         return loss

#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels)           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}

#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

#     def epoch_end(self, epoch, result):
#         print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '', 
#             result['train_loss'], result['val_loss'], result['val_acc']))

class Model(FModule):
    def __init__(self, in_channel=3, num_classes=100):
        super().__init__()
        # Use a pretrained model
        weights = ResNet18_Weights.DEFAULT
        self.network = resnet18(weights=weights)
        # Replace last layer
        self.network.fc = nn.Linear(self.network.fc.in_features, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.network(x))
        out = self.classifier(out)
        return out
    
    def pred_and_rep(self,x):
        e = F.relu(self.network(x))
        o = self.classifier(e)
        return o, e