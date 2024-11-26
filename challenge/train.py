import torch
import torchvision
import utils # custom utility funcions
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from PIL import Image



path = '/home/rick/Ri/ThirdYear/DeepLearning/challenge/dl2425_challenge_dataset/'
model_name = "challenge.pt"
train_path = path + "train"
val_path = path + "val"
train_datafolder = torchvision.datasets.ImageFolder(root=train_path)
val_datafolder = torchvision.datasets.ImageFolder(root=val_path)

print(len(val_datafolder.samples))
print(len(train_datafolder.samples))
