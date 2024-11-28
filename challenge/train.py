import torch
import torchvision
# import utils # custom utility funcions
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from PIL import Image
from MLP import MLP
from utils import training_loop
from utils import validate
import datetime


path = '/home/rick/Ri/ThirdYear/DeepLearning/challenge/dl2425_challenge_dataset/'
model_name = "challenge.pt"
train_path = path + "train"
val_path = path + "val"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformations = transforms.Compose([
    transforms.ToTensor()
    #transforms.Grayscale(),
    #transforms.GaussianBlur(5) # 5x5 kernel gaussian blur
])
train_datafolder = torchvision.datasets.ImageFolder(root=train_path, transform=transformations)
val_datafolder = torchvision.datasets.ImageFolder(root=val_path, transform=transformations)


print(f"validation: {len(val_datafolder.samples)}")
print(f"training: {len(train_datafolder.samples)}")

train_imgs = {c:[] for c, x in enumerate(train_datafolder.classes)}
val_imgs = {c:[] for c, x in enumerate(val_datafolder.classes)}

n_samples = len(train_datafolder)
train_shuffled_indices = torch.randperm(n_samples)
train_indices = train_shuffled_indices[:int(n_samples*.3)]
train_data = [train_datafolder[x] for x in train_indices]
del train_datafolder

n_samples = len(val_datafolder)
val_shuffled_indices = torch.randperm(n_samples)
val_indices = val_shuffled_indices[:int(n_samples*.5)]
val_data = [val_datafolder[x] for x in val_indices]
del val_datafolder

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)

val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = True)

model = MLP()

numel_list = [p.numel() for p in model.parameters()]
print("[*] Number of parameters:", sum(numel_list), numel_list)

optimizer = optim.SGD(model.parameters(), lr=.6e-2, weight_decay=1e-3) # NOTE: weight_decay acts like l2 regularization
loss_fn = nn.CrossEntropyLoss()
n_epochs = 60

# train the model
print(f"[*] TRAINING for {n_epochs} epochs")
training_loop(
     n_epochs = n_epochs,
     optimizer = optimizer,
     model = model,
     loss_fn = loss_fn,
     train_loader = train_loader,
     val_loader = val_loader
    )

