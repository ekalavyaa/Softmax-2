import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np


class DatasetCreate(Dataset):
    def __init__(self):
        data =np.loadtxt('data/train.csv', delimiter=",", skiprows=1, dtype=np.float)
        self.len = data.shape[0]
        self.train_x = data[:,0:-1]
        self.train_y = data[:,[-1]]

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]    

    def __len__ (self):
        return self.len

