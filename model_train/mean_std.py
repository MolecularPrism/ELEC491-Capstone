import numpy as np

train_X = np.load("./data/train/X.npy")  # (N,T,C)

mean = train_X.mean(axis=(0, 1))
std  = train_X.std(axis=(0, 1))

print("mean:", mean)
print("std :", std)
