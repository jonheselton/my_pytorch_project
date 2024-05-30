import random, string, os, torch, torchvision
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
# Used from UNet2DModel from diffusers to understand the size changes in the Down and Up layers
def corrupt_original(x, amount):
  """Corrupt the input `x` by mixing it with noise according to `amount`"""
  noise = torch.rand_like(x).to('cuda')
  amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
  return x*(1-amount) + noise*amount 

# Experimenting with alternative corruption functions
def add_noise_gaussian(image, noise_level):
# Adds Gaussian noise to an image tensor.
  noise = torch.randn(image.shape).to('cuda') * noise_level
  return image + noise

def add_noise_salt_and_pepper(image, noise_prob):
#  Adds salt-and-pepper noise to an image tensor. Probablity a pixel is corrupted
  salt_value = 1.0
  pepper_value = 0.0
  noise_mask = torch.empty_like(image).uniform_(0, 1).to('cuda')
  salt_mask = noise_mask < noise_prob * 0.5
  pepper_mask = noise_mask >= noise_prob * 0.5
  image = image.clone().to('cuda')
  image[salt_mask] = salt_value
  image[pepper_mask] = pepper_value
  return image

def random_augment(image):
# Randomly does Gaussian or SnP
  augmentation_type = random.choice(['gaussian', 'salt_and_pepper'])
  if augmentation_type == 'gaussian':
      noise_level = random.uniform(0.01, 0.2)  # Adjust noise level range
      return add_noise_gaussian(image, noise_level)
  else:
      noise_prob = random.uniform(0.01, 1.0)  # Adjust noise probability range
      return add_noise_salt_and_pepper(image, noise_prob)


def corrupt_guassian(images, noise_modifier = 0.0):
    noisy_images = []
    noise_level = random.uniform(0.02 + noise_modifier, 0.06 + noise_modifier)  # Adjust noise level range ~0.1 is where it becomes noticable to me
    # noise_level = 0.8
    for image in images:
        noisy_images.append(add_noise_gaussian(image.clone(), noise_level))
        noise_level += 0.02
    return torch.stack(noisy_images).to('cuda')

def corrupt(images):
  noisy_images = []
  for image in images:
      noisy_images.append(random_augment(image.clone()))
  return torch.stack(noisy_images).to('cuda')
def images_to_tb(images, tag = 'name_tag'):
    # Send a NCHW tensor to tensorboard
    writer = SummaryWriter(f'logs/{randomword(5)}')
    writer.add_images(tag, images)

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
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
 #           nn.BatchNorm2d(256),
 #           nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
 #           nn.BatchNorm2d(512),
 #           nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
 #           nn.BatchNorm2d(512),
 #           nn.LeakyReLU(inplace=True),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
#            nn.BatchNorm2d(512),
#            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
#            nn.BatchNorm2d(256),
#            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.LeakyReLU() # The activation function
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)
    
    # def forward(self, x):
    #     h = []
    #     for i, l in enumerate(self.down_layers):
    #         x = l(x) # Convolution -> BN -> Activate
    #         if isinstance(i, nn.LeakyReLU) and i < 6:
    #           h.append(x) # Storing output for skip connection
    #           x = self.downscale(x) # Downscale ready for the next layer
    #     for i, l in enumerate(self.up_layers):
    #         if isinstance(i, nn.Conv2d) and i > 2: # For all except the first up layer
    #           x = self.upscale(x) # Upscale
    #           x += h.pop() # Fetching stored output (skip connection)
    #         x = l(x)
    #     return x

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
device = 'cuda'
run_id = f'{os.path.basename(__file__)}'.strip('.py') + f'_{randomword(5)}'
print(f'Beginning training run {run_id}')
writer = SummaryWriter(f'logs/{run_id}') 
dataroot = "data/celeba"
workers = 16
batch_size = 32
image_size = 256

n_epochs = 3

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

opt = torch.optim.AdamW(net.parameters(), lr=1e-4) 

running_loss = 0.0
i = 0
pbar = tqdm(total=len(train_dataloader) * n_epochs)
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        # Get some data and prepare the corrupted version
        x = x.to(device)
        noise_modifier = (i//1800)*.0225
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
        if i % 500 == 499:
            img_stack_0 = torch.stack((x[-1], noisy_x[-1], pred[-1]))
            writer.add_images('Image Sampls', img_stack_0, i)
        i += 1
        pbar.update(1)
        
    PATH = f'models/{run_id}_{epoch}.pth'
    torch.save(net.state_dict(), PATH)
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
        pred = net(x)
        q = net(q)
    mix_factor = 1/(n_steps - i)
    x = x*(1-mix_factor) + pred*mix_factor
    if i % 5 == 0:
        for j, img in enumerate(pred): # j = num of fake images
            save_image(img, f'generated_images/{run_id}/celeb_step_{i}_img_{j}.png')
        for j, img in enumerate(q): # j = num of fake images
            save_image(img, f'generated_images/{run_id}/qceleb_step_{i}_img_{j}.png')
save_image(q, f'generated_images/{run_id}/q.png')