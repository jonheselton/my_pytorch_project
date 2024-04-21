#!python

import torch
import numpy as np

# PyTorch - Autograd
# Neural Networks (NN) are made up of nested functions.  The functions are defined by weighs and biases
# Weight - A value that is output (learned) during the training process, and used as an input when making predictions
# Bias (mathematical) - a value indicating the offset from the origin.  In a 2D curve, it is the value of Y when X = 0
# Forward Propegation - Running an input through the NN to arrive at a prediction
# Backward Propegation - Taking the prediction and determining how to adjust the weights and bias to make a more accurate prediction.abs
# Learning rate - A float that controls how much parameters are changed by during gradient descent.  Too low and the model takes too long, too high and convergence may not be reached
# Convergence - The point at which changes in error become so small as to be negligible
# Momentum - A gradient descent algo that takes into account the derivitives of preceeding steps in addition to the current step.
# Gradient descent - A mathematical technique to minimize loss. Gradient descent iteratively adjusts weights and biases, gradually finding the best combination to minimize loss.


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

