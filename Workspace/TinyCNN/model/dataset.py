import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class KFallDataset(Dataset):
    """
    read data/{train,val}/X.npy, y.npy
    return (x, y), where:
      - x: (C, T) float32
      - y: long
    """
    def __init__(self, data_dir, channels_first=True, normalize=None):
        self.X = np.load(os.path.join(data_dir, "X.npy"))  # (N, T, C)
        self.y = np.load(os.path.join(data_dir, "y.npy"))

        self.X = self.X.astype(np.float32)
        self.y = self.y.astype(np.int64)

        self.channels_first = channels_first
        self.normalize_stats = normalize  # {"mean":..., "std":...}

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, C)
        y = self.y[idx]

        # normalize
        if self.normalize_stats is not None:
            mean = self.normalize_stats["mean"]
            std = self.normalize_stats["std"]
            std = np.where(std == 0, 1.0, std)
            x = (x - mean) / std

        if self.channels_first:
            x = x.T  # (C, T)

        return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)


def get_loaders(root_dir="./data", batch_size=128, normalize=True):
    """
    from data/train and data/val build dataloader
    """
    # first load training set to calculate normalization stats
    train_X = np.load(os.path.join(root_dir, "train", "X.npy"))
    mean = train_X.mean(axis=(0, 1))  # (C,)
    std = train_X.std(axis=(0, 1))
    norm_stats = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)} if normalize else None

    train_set = KFallDataset(os.path.join(root_dir, "train"), normalize=norm_stats)
    val_set   = KFallDataset(os.path.join(root_dir, "val"), normalize=norm_stats)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_loaders("./data", batch_size=64)

    for xb, yb in train_loader:
        print("xb:", xb.shape)  # [B, 6, 50]
        print("yb:", yb.shape)  # [B]
        break
