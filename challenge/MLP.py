import torch
import torch.nn as nn
import torch.nn.init as init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, c=1, h=224, w=224, nout=2):
        # b = batch size
        # c = channels
        # h = height
        # w = widthb
        super(MLP, self).__init__()
        s0 = c*h*w
        s1 = s0 // 2
        s2 = s1 // 2
        self.fc1 = nn.Linear(s0, s1)
        self.fc2 = nn.Linear(s1, s2)
        self.fc3 = nn.Linear(s2, nout)
    def init_params(self):
        with torch.no_grad():
            init.xavier_uniform_(self.fc1.weight)
            init.xavier_uniform_(self.fc2.weight)
            init.xavier_uniform_(self.fc3.weight)
            init.zeros_(self.fc1.bias)
            init.zeros_(self.fc2.bias)
            init.zeros_(self.fc3.bias)
    def forward(self, X):
        # TODO: Add normalization layers
        # TODO: Change activation function for last layer?
        X = X.view(X.size(0), -1) # Flatten
        out = torch.relu(self.fc1(X))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        return out

