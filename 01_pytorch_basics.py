#Importing required dependencies
import torch
import numpy as np


# Creating Tensors
t1 = torch.tensor([1.0, 2.0, 3.0])
print(t1)
print(t1.shape)
print(t1.dtype)

t2 = torch.tensor(([1, 2], [3,4], [5,6]))
print(t2)
print(t2.shape)
print(t2.dtype)