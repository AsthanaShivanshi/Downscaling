import torch
import pandas as pd
from collections import defaultdict

def split_by_decade(times, seed=42, train_frac=0.7, val_frac=0.2):
    torch.manual_seed(seed)
    times = pd.to_datetime(times)
    decade_to_indices = defaultdict(list)

    for idx, t in enumerate(times):
        decade = int(t.year // 10 * 10)
        decade_to_indices[decade].append(idx)

    train_indices, val_indices, test_indices = [], [], []

    for indices in decade_to_indices.values():
        indices = torch.tensor(indices)
        indices = indices[torch.randperm(len(indices))]

        n_total = len(indices)
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        n_test = n_total - n_train - n_val #Everything else is used for testing the model. Should I change to everything else used for training? SHALL SEE . UNDECIDED NOW

        train_indices.append(indices[:n_train])
        val_indices.append(indices[n_train:n_train + n_val])
        test_indices.append(indices[n_train + n_val:])

    return torch.cat(train_indices), torch.cat(val_indices), torch.cat(test_indices)
