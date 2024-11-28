import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, c=3, h=224, w=224, nout=2, nchans1=16):
        super(MLP, self).__init__()

        self.nchans1 = nchans1

        # first convolution and batch normalization
        self.conv1 = nn.Conv2d(3, nchans1, kernel_size=3, padding=1)
        self.conv1_batchnorm = nn.BatchNorm2d(num_features=nchans1)

        # second convolution and batch normalization
        self.conv2 = nn.Conv2d(nchans1, nchans1 // 2, kernel_size=3, padding=1)
        self.conv2_batchnorm = nn.BatchNorm2d(num_features=nchans1 // 2)

        # third convolution and batch normalization
        self.conv3 = nn.Conv2d(nchans1 // 2, nchans1 // 4, kernel_size=3, padding=1)
        self.conv3_batchnorm = nn.BatchNorm2d(num_features=nchans1//4)

        # fully connected layers
        self.s1 = 4 * 28 * 28
        self.fc1 = nn.Linear(self.s1, 32)
        self.fc2 = nn.Linear(32, nout)
    def init_params(self):
        with torch.no_grad():
            init.xavier_uniform_(self.fc1.weight)
            init.xavier_uniform_(self.fc2.weight)
            init.xavier_uniform_(self.fc3.weight)
            init.xavier_uniform_(self.conv1.weight)
            init.xavier_uniform_(self.conv2.weight)
            init.xavier_uniform_(self.conv3.weight)
            init.zeros_(self.fc1.bias)
            init.zeros_(self.fc2.bias)
            init.zeros_(self.fc3.bias)
    def forward(self, X):
        # TODO: Add normalization layers
        # TODO: Change activation function for last layer?
        # X = X.view(X.size(0), -1) # Flatten
        # print(X.shape)
        out = F.max_pool2d(torch.relu(self.conv1_batchnorm(self.conv1(X))), 2)
        # print(out.shape)
        out = F.max_pool2d(torch.relu(self.conv2_batchnorm(self.conv2(out))), 2)
        # print(out.shape)
        out = F.max_pool2d(torch.relu(self.conv3_batchnorm(self.conv3(out))), 2)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

