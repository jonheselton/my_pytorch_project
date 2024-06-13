import random, string, os, torch, torchvision, time
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np
import os

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
           nn.Conv2d(in_channels, 128, kernel_size=5, padding=2),
           nn.BatchNorm2d(128),
           nn.LeakyReLU(inplace=True),
           nn.Conv2d(128, 256, kernel_size=5, padding=2),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(inplace=True),
           nn.Conv2d(256, 256, kernel_size=5, padding=2),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(inplace=True),
        ])
        self.up_layers = torch.nn.ModuleList([
           nn.Conv2d(256, 256, kernel_size=5, padding=2),
           nn.BatchNorm2d(256),
           nn.LeakyReLU(inplace=True),
           nn.Conv2d(256, 128, kernel_size=5, padding=2),
           nn.BatchNorm2d(128),
           nn.LeakyReLU(inplace=True),
           nn.Conv2d(128, out_channels, kernel_size=5, padding=2), 
        ])
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    
    def forward(self, x):
        h = []
        # Down!
        for i, l in enumerate(self.down_layers):
            x = l(x) # Convolution -> BN -> Activate
            if isinstance(i, nn.LeakyReLU) and i < 6:
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer
        # Back Up
        for i, l in enumerate(self.up_layers):
            if isinstance(i, nn.Conv2d) and i > 2: # For all except the first up layer
              x = self.upscale(x) # Upscale
              x += h.pop() # Fetching stored output (skip connection)
            x = l(x)
        return x
device = 'cuda'



def corrupt(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x)
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount 


run_id = 'V4-1_wuwqz_1000steps'
PATH = 'models/V4-1_wuwqz-final.pth'
net = BasicUNet().to(device)
net.load_state_dict(torch.load(PATH))
# Sampling
n_steps = 1000
x = torch.rand(4, 3, 128, 128).to(device)
q = x
for i in range(n_steps):
    save_image(x, f'generated_images/{run_id}/xceleb_step_{i}_img.png')
    with torch.no_grad():
        pred = net(x)
        q = net(q)
    # Go back and get a better understanding of this
    mix_factor = 1/(n_steps - i)
    x = x*(1-mix_factor) + pred*mix_factor
    save_image(pred, f'generated_images/{run_id}/celeb_step_{i}_img.png')
    save_image(q, f'generated_images/{run_id}/qceleb_step_{i}_img.png')
save_image(q, f'generated_images/{run_id}/q.png')


