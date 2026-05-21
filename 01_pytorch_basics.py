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


# Random Tensors
random_t = torch.randn(3,4)
print(random_t)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
t1 = t1.to(device)
print(f'Tensor is on: {t1.device}')



#----------------------------------------------#

# Gradient Descent

x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x + 1

y.backward()     # calculate the derivative of tensor x
print(x.grad)  


w = torch.randn(1, requires_grad=True)
loss = (w-5)**2
loss.backward()
print(w.grad)

w.data -= 0.1 * w.grad
print(w.data)


