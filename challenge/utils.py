import torch
import math
import numpy as np
from PIL import Image

def train_val_test_split(data):
    n_samples = len(data)
    n_val = int(0.3 * n_samples)
    n_test = int(0.1 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:-n_test]
    test_indices = shuffled_indices[-n_test:]
    #print(train_indices.shape, val_indices.shape, test_indices.shape, sep="\n")

    train_data = torch.stack([data[x] for x in train_indices])
    val_data = torch.stack([data[x] for x in val_indices])
    test_data = torch.stack([data[x] for x in test_indices])

    assert len(train_data) == train_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (train data)"
    assert len(val_data) == val_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (val data)"
    assert len(test_data) == test_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (test data)"

    return train_data, val_data, test_data

def is_full_rank(X:torch.tensor) -> bool:
    '''A matrix X is full rank if no column feature is linearly independent on the others.'''
    return X.shape[1] == torch.linalg.matrix_rank(X)
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma**2)

def show_img(path:str) -> None:
    img = Image.open(path)
    path.show()

def show_tensor(t_img:torch.tensor) -> None:
    if len(t_img.shape > 3):
        assert t_img.shape[0] <= 1, "You're trying to display a batch of size > 1. Be wary of opening more than one image at a time..."
    to_img = transforms.ToPILImage()
    img = to_img(t_img)
    img.show()

def training_loop(n_epochs, optimizer, model, loss_fn,
                        train_loader) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        if epoch == 1 or epoch % 1 == 0:
            print(f"{datetime.datetime.now()} Epoch {epoch}, Training loss {loss_train / len(train_loader)}")
    return



