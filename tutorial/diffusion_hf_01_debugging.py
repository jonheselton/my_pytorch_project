import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
from torchvision.utils import save_image

# Used from UNet2DModel from diffusers to understand the size changes in the Down and Up layers

def corrupt(x, amount):
    for i in range(amount + 1):
        noise = torch.randn_like(x)
        noisey = x + noise
    return noisey

class BasicUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = torch.nn.ModuleList([ 
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.Conv2d(256, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.ReLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x)) # Through the layer and the activation function
            # print(f'Size of x after down layer {i} activation {x.size()}')
            if i < 2: # For all but the third (final) down layer:
            #   print(f'Down Layers being added to h in loop {i} {x.size()}')
              h.append(x) # Storing output for skip connection
              x = self.downscale(x) # Downscale ready for the next layer
            #   print(f'Size of x after down downscale in layet {i} {x.size()} prepared for next layer')
        for i, l in enumerate(self.up_layers):
            if i > 0: # For all except the first up layer
              x = self.upscale(x) # Upscale
            #   print(f'Up layers Loop {i} x size {x.size()} and in h {h[-1].size()}')
              x += h.pop() # Fetching stored output (skip connection)
            x = self.act(l(x)) # Through the layer and the activation function
            # print(f'x after activation in up layer {i} {x.size()}')
        return x
device = torch.device("cuda")
dataroot = "data/celeba"
workers = 16
batch_size = 16
image_size = 256
model_path = 'models/subsets_unet_diffusion.pth'
loss_fn = nn.SmoothL1Loss()

dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

subset = Subset(dataset, range(0,16*16))
data = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=workers)
net = BasicUNet().to(device)
net.load_state_dict(torch.load(model_path))
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 

x, y = next(iter(subset))
x = x.to(device)
noise_amount = 1
noisy_x = corrupt(x, noise_amount)
pred = net(noisy_x)
loss = loss_fn(pred, x)
print(loss.item())

# _noise_amount = torch.rand(x.shape[0]).to(device)
# _noise = torch.rand_like(x)
# _amount = amount.view(-1, 1, 1, 1)
# _noisy_x = x*(1-amount) + noise*amount


# noise = torch.randn_like(x)
# noise_2 = torch.randn_like(x)
# noisey = x + noise
# noisey_2 = noisey + noise_2
# output = [x, noise, noisey, noisey_2]



