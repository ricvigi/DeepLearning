import torch
def train_val_test_split(data):
    n_samples = len(data)
    n_val = int(0.3 * n_samples)
    n_test = int(0.1 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:-n_test]
    test_indices = shuffled_indices[-n_test:]
    print(train_indices.shape, val_indices.shape, test_indices.shape, sep="\n")

    train_data = [data[x] for x in train_indices]
    assert len(train_data) == train_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (train data)"
    val_data = [data[x] for x in val_indices]
    assert len(val_data) == val_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (val data)"
    test_data = [data[x] for x in test_indices]
    assert len(test_data) == test_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (test data)"
    return train_data, val_data, test_data
