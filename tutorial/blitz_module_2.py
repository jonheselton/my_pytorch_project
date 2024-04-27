#!python

import torch
import numpy as np

# PyTorch - Autograd

# Note - This example will not run on a gpu.
#      The labels for this model has a shape of (1,1000)

import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data) # forward pass

loss = (prediction - labels).sum() # calculate the error
loss.backward() # backward pass

# Load the model's parameters in to the optimizer, set the learning rate (lr) and momentum
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# Finally, the optimizer adjusts each parameter by its gradient
optim.step() #gradient descent

