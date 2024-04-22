# Missed module 3, so doing it after i finished 4

# affine operation
# channel
# convolution
# RELU activation function
# connections
# Fully connected
# Guasian layer
# Loss Function - Calculates loss, or error, based on the output of the nn and the target.  torch.nn has many waiting to be used

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))
        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        # Convolution layer C3: 6 input channels, 16 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))
        # Subsampling layer S4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a (N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)
        # Flatten operation: purely functional, outputs a (N, 400) Tensor
        s4 = torch.flatten(s4, 1)
        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))
        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))
        # Gaussian layer OUTPUT: (N, 84) Tensor input, and
        # outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


net = Net()
print(net)

# Calculating Loss
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss() # sets the loss function

loss = criterion(output, target) # loss function acts on the output and target
print(loss) # returns a tensor

# This helps visualize the forward function leading to the output, and finally the loss.
# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> flatten -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss

# When we step backwards through the nn, we move up one function at a time, updating the gradient of each step
# To visualize...
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Run the back propegation
loss.backward()

# Now that backpropegation has been run, and the gradients are set, we need to adjust the weights
# The simplest weight optimization is weight = weight - learning_rate * gradient
# weights are a parameter of the nn

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# There are many different rules for changing the parameters.  the nn module has some built in
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01) # input the parameters and learning rate (lr) into the optimizer

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers, we don't want the gradients to accumulate during the backpro
output = net(input) # gets the output of the nn
loss = criterion(output, target) # calculates the loss
loss.backward() # propegate the gradient back through the nn
optimizer.step()    # Update the weights!

