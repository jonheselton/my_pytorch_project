import torch
from torch import nn

class BasicConvolutionUNet(nn.Module):
    """A minimal UNet implementation."""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down_layers = nn.ModuleList([ 
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
