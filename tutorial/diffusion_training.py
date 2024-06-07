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

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
# Experimenting with alternative corruption functions
def add_noise_gaussian(image, noise_level):
# Adds Gaussian noise to an image tensor.
  noise = torch.randn(image.shape).to('cuda') * noise_level
  return image + noise
def corrupt_guassian(images, noise_modifier = 0.0):
    noisy_images = []
    noise_level = random.uniform(0.4 + noise_modifier, 0.6 + noise_modifier)  # Adjust noise level range ~0.1 is where it becomes noticable to me
    # noise_level = 0.8
    for image in images:
        noisy_images.append(add_noise_gaussian(image.clone(), noise_level))
    return torch.stack(noisy_images).to('cuda')
class BasicUNet(nn.Module):
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
run_id = f'{os.path.basename(__file__)}'.strip('.py') + f'_{randomword(5)}'
print(f'Beginning training run {run_id}')
writer = SummaryWriter(f'logs/{run_id}') 
dataroot = "data/celeba"
workers = 16
batch_size = 128
image_size = 128
n_epochs = 5
loss_fn = nn.SmoothL1Loss(beta=1.0) 
# Use celeb dataloader instead
# Create the network
net = BasicUNet()
PATH = 'models/diffusion_training_imvmr-final.pth'
net.load_state_dict(torch.load(PATH))
net.to(device)
opt = torch.optim.AdamW(net.parameters(), lr=1e-4) 
# projection(writer, dataset, 250)
# model_graph(writer, net, dataset)
running_loss = 0.0
i = 0
pbar = tqdm(total=1583 * n_epochs)
for epoch in range(n_epochs):
    dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.RandomGrayscale(),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
#                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    for x, y in train_dataloader:
        # Get some data and prepare the corrupted version
        x = x.to(device)
        noise_modifier = (i//527)*.05
        noisy_x = corrupt_guassian(x, noise_modifier).to(device) # Create our noisy x
        # Get the model prediction
        pred = net(noisy_x)
        # Calculate the loss
        loss = loss_fn(pred, x).to(device) # How close is the output to the true 'clean' x?
        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        # Log weights and gradients
        for name, param in net.named_parameters():
            writer.add_histogram(f"weights/{name}", param.data, i)
            writer.add_histogram(f"gradients/{name}", param.grad.data, i)
        opt.step()
        running_loss += loss.item()
        if i % 250 == 249:    # every 250 mini-batches...
            writer.add_scalar('training loss', running_loss / 250, i)
            running_loss = 0.0
            img_stack_0 = torch.stack((x[-3], noisy_x[-3], pred[-3]))
            writer.add_images('Image Sampls', img_stack_0, i)
        i += 1
        pbar.update(1)
        
pbar.close()
writer.close()
print(f'Training for model id {run_id} completed')
PATH = f'models/{run_id}-final.pth'
torch.save(net.state_dict(), PATH)

