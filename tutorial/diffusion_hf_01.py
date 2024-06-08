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
def save_images_to_tb(tensor_stack, writer, n):
    # create grid of images
    img_grid = torchvision.utils.make_grid(img_stack)
    # write to tensorboard
    writer.add_image(f'noise_demo', img_grid, n)

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
def add_noise_gaussian(image, noise_level):
# Adds Gaussian noise to an image tensor.
  noise = torch.randn(image.shape).to('cuda') * noise_level
  return image + noise

def corrupt_guassian(images, noise_modifier = 0.0):
    noisy_images = []
    noise_level = random.uniform(0.4 + noise_modifier, 0.8 + noise_modifier)  # See noise_level tensorboard for examples
    for image in images:
        noisy_images.append(add_noise_gaussian(image.clone(), noise_level))
    return torch.stack(noisy_images).to('cuda'), noise_level

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
run_id = f'V3_{os.path.basename(__file__)}'.strip('.py') + f'_{randomword(5)}'
print(f'Beginning training run {run_id}')
writer = SummaryWriter(f'logs/{run_id}') 
dataroot = "data/celeba"
workers = 16
batch_size = 128
image_size = 128

n_epochs = 10

loss_fn = nn.SmoothL1Loss(beta=1.0) 
# Use celeb dataloader instead
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
# Create the network
net = BasicUNet()
net.to(device)
initialize_weights(net)

opt = torch.optim.AdamW(net.parameters(), lr=2e-4) 
# projection(writer, dataset, 250)
# model_graph(writer, net, dataset)
running_loss = 0.0
i = 0
pbar = tqdm(total=len(train_dataloader) * n_epochs)
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        # Get some data and prepare the corrupted version
        x = x.to(device)
        noise_modifier = 0.1 * epoch
        noisy_x, noise_level = corrupt_guassian(x, noise_modifier) # Create our noisy x
        # Get the model prediction
        pred = net(noisy_x)
        # Calculate the loss
        loss = loss_fn(pred, x).to(device) # How close is the output to the true 'clean' x?
        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        # Log noise level
        writer.add_scalar('noise', noise_level, i)
        # Log weights and gradients
        for name, param in net.named_parameters():
            writer.add_histogram(f"weights/{name}", param.data, i)
            writer.add_histogram(f"gradients/{name}", param.grad.data, i)
        opt.step()
        running_loss += loss.item()
        if i % 250 == 249:    # every 250 mini-batches...
            writer.add_scalar('training loss', running_loss / 250, i)
            running_loss = 0.0
            img_stack_0 = torch.stack((x[-1], noisy_x[-1], pred[-1]))
            writer.add_images('Image Sampls', img_stack_0, i)
        i += 1
        pbar.update(1)
        
pbar.close()
writer.close()
print(f'Training for model id {run_id} completed')
PATH = f'models/{run_id}-final.pth'
torch.save(net.state_dict(), PATH)
# PATH = 'models/diffusion_hf_01_ocu.pth'
# net = BasicUNet().to(device)
# net.load_state_dict(torch.load(PATH))
# Sampling
n_steps = 100
x = torch.rand(4, 3, 256, 256).to(device)
q = x
os.makedirs(f'generated_images/{run_id}')
for i in range(n_steps):
    with torch.no_grad():
        pred, p = net(x)
        q, p = net(q)
    # Go back and get a better understanding of this
    mix_factor = 1/(n_steps - i)
    x = x*(1-mix_factor) + pred*mix_factor
    if i % 5 == 0:
        for j, img in enumerate(pred): # j = num of fake images
            save_image(img, f'generated_images/{run_id}/celeb_step_{i}_img_{j}.png')
        for j, img in enumerate(q): # j = num of fake images
            save_image(img, f'generated_images/{run_id}/qceleb_step_{i}_img_{j}.png')
save_image(q, f'generated_images/{run_id}/q.png')
