import torch


def train_val_test_split(data):
    n_samples = len(data)
    n_val = int(0.3 * n_samples)
    n_test = int(0.1 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:-n_test]
    test_indices = shuffled_indices[-n_test:]
    #print(train_indices.shape, val_indices.shape, test_indices.shape, sep="\n")

    train_data = [data[x] for x in train_indices]
    val_data = [data[x] for x in val_indices]
    test_data = [data[x] for x in test_indices]

    assert len(train_data) == train_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (train data)"
    assert len(val_data) == val_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (val data)"
    assert len(test_data) == test_indices.shape[0], "[*] ERROR: Something went wrong in splitting data (test data)"

    return train_data, val_data, test_data

def is_full_rank(X:torch.tensor) -> bool:
    '''A matrix X is full rank if no column feature is linearly independent on the others.'''
    return X.shape[1] == torch.linalg.matrix_rank(X)
