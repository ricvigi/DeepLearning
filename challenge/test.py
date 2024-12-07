import torch
import torchvision
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from PIL import Image
from MLP import MLPAlex
from utils import training_loop
from utils import validate
from custom_dataset import TestDataset
import datetime

gpath = "/home/rick/Ri/ThirdYear/DeepLearning/challenge/"
path = '/home/rick/Ri/ThirdYear/DeepLearning/challenge/dl2425_challenge_dataset/'
model_name = "challenge_alex.pt"
test_path = path + "test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformations = transforms.Compose([
    transforms.ToTensor()
    #transforms.Grayscale(),
    #transforms.GaussianBlur(5) # 5x5 kernel gaussian blur
])
test_datafolder = TestDataset(folder_path=test_path,
                                                    transform=transformations)
# print(f"test: {len(test_datafolder.samples)}")
test_loader = torch.utils.data.DataLoader(test_datafolder, batch_size = 64, shuffle = False, num_workers=8)

model = MLPAlex()

if os.path.exists(gpath + model_name):
    print(f"[*] Loading model's state dict {model_name}")
    model.load_state_dict(torch.load(gpath + model_name, weights_only=True, map_location=torch.device("cpu")))

numel_list = [p.numel() for p in model.parameters()]
print("[*] Number of parameters:", sum(numel_list), numel_list)

model.eval()
