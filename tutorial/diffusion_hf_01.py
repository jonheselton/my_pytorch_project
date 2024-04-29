import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
#from matplotlib import pyplot as plt
import torchvision.datasets as dset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torchvision.utils import save_image

device = torch.device("cuda")
#print(f'Using device: {device}')

def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount 

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.SiLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            print(f'down {i} {x.shape}')
            if i < 2: # For all but the third (final) down layer:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer
  
        for i, l in enumerate(self.up_layers):
            print(f'{x.shape} before uo loop {i}')
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
              print(f'shape of x {x.size()} after upscale {i}')
              print(f'shape of h[-1] {h[-1].size()} after upscale {i}')
              try:
                x += h.pop() # Fetching stored output (skip connection)
            
              print(f'shape of x {x.size()} after upscale and h-pop {i}')
            x = self.act(l(x)) # Through the layer and the activation function
            #print(f'{type(x)} on loop {i} after activation')
            
            
        return x
dataroot = "data/celeba"
workers = 16
batch_size = 16
image_size = 256
# Use celeb dataloader instead
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)


# Dataloader (you can mess with batch size)
#batch_size = 128
#train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# How many runs through the data should we do?
n_epochs = 3

# Create the network
net = BasicUNet()
net.to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 

# Keeping a record of the losses for later viewing
losses = []

# The training loop
# for epoch in range(n_epochs):

#     for y, x in enumerate(train_dataloader, 0):
#         # Get some data and prepare the corrupted version
#         x = x[0].to(device) # Data on the GPU
#         noise_amount = torch.rand(x.shape[0]).to(device) # Pick random noise amounts
#         noisy_x = corrupt(x, noise_amount) # Create our noisy x

#         # Get the model prediction
#         pred = net(noisy_x)

#         # Calculate the loss
#         loss = loss_fn(pred, x) # How close is the output to the true 'clean' x?

#         # Backprop and update the params:
#         opt.zero_grad()
#         loss.backward()
#         opt.step()

#         # Store the loss for later
#         losses.append(loss.item())

#     # Print our the average of the loss values for this epoch:
#     avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
#     print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

# Save the model
PATH = './models/unet_diffusion.pth'
# torch.save(net.state_dict(), PATH)
net = BasicUNet().to(device)
net.load_state_dict(torch.load(PATH))
# Sampling
n_steps = 40
x = torch.rand(32, 3, 5, 5).to(device)

for i in range(n_steps):
    noise_amount = torch.ones((x.shape[0], )).to(device) * (1-(i/n_steps)) # Starting high going low
    with torch.no_grad():
        pred = net(x)
    mix_factor = 1/(n_steps - i)
    x = x*(1-mix_factor) + pred*mix_factor
    for img in enumerate(pred): # j = num of fake images
        save_image(img, f'generated_images/celeb_step_{i}.png')