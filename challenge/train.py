import torch
import torchvision
import os
import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from PIL import Image
from MLP import MLP
from MLP import MLPAlex
from utils import training_loop
from utils import validate


gpath = "/home/rick/Ri/ThirdYear/DeepLearning/challenge/"
path = '/home/rick/Ri/ThirdYear/DeepLearning/challenge/dl2425_challenge_dataset/'
model_name = "challenge.pt"
train_path = path + "train"
val_path = path + "val"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_num_threads(4)
# torch.set_num_threads(8) # try with 8
transformations = transforms.Compose([
    transforms.ToTensor()
    #transforms.Grayscale(),
    #transforms.GaussianBlur(5) # 5x5 kernel gaussian blur
])
train_datafolder = torchvision.datasets.ImageFolder(root=train_path,
                                                    transform=transformations)
val_datafolder = torchvision.datasets.ImageFolder(root=val_path,
                                                  transform=transformations)

print(f"validation: {len(val_datafolder.samples)}")
print(f"training: {len(train_datafolder.samples)}")

# train_imgs = {c:[] for c, x in enumerate(train_datafolder.classes)}
# val_imgs = {c:[] for c, x in enumerate(val_datafolder.classes)}

n_samples = len(train_datafolder)
train_shuffled_indices = torch.randperm(n_samples)
train_indices = train_shuffled_indices[:int(n_samples*1)]
train_data = [train_datafolder[x] for x in train_indices]
del train_datafolder

n_samples = len(val_datafolder)
val_shuffled_indices = torch.randperm(n_samples)
val_indices = val_shuffled_indices[:int(n_samples*1)]
val_data = [val_datafolder[x] for x in val_indices]
del val_datafolder

train_loader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True, num_workers=8)

val_loader = torch.utils.data.DataLoader(val_data, batch_size = 64, shuffle = True, num_workers=8)

model = MLP()

numel_list = [p.numel() for p in model.parameters()]
print("[*] Number of parameters:", sum(numel_list), numel_list)

# model.load_state_dict() needs weights_only = True for security
# reasons. Leaving the default value (False) could lead to
# arbitrary execution of code
if os.path.exists(gpath + model_name):
    print(f"[*] Loading model's state dict {model_name}")
    model.load_state_dict(torch.load(gpath + model_name, weights_only=True))



# optimizer = optim.SGD(model.parameters(), lr=.5e-3, weight_decay=1e-2) # NOTE: weight_decay acts like l2 regularization
optimizer = optim.Adam(model.parameters(), lr = .5e-3, weight_decay=1e-2)
loss_fn = nn.CrossEntropyLoss()


n_epochs = 5

# train the model
print(f"[*] TRAINING for {n_epochs} epochs")
training_loop(
     n_epochs = n_epochs,
     optimizer = optimizer,
     model = model,
     loss_fn = loss_fn,
     train_loader = train_loader,
     val_loader = None
    )

# update the model's state dictionary
torch.save(model.state_dict(), gpath + model_name)
