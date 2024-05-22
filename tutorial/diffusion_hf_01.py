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
            nn.Conv2d(in_channels, 256, kernel_size=5, padding=2),
            nn.Conv2d(256, 512, kernel_size=5, padding=2),
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
        ])
        self.up_layers = torch.nn.ModuleList([
            nn.Conv2d(512, 512, kernel_size=5, padding=2),
            nn.Conv2d(512, 256, kernel_size=5, padding=2),
            nn.Conv2d(256, out_channels, kernel_size=5, padding=2), 
        ])
        self.act = nn.LeakyReLU() # The activation function
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
run_id = f'{os.path.basename(__file__)}'.strip('.py') + f'_{randomword(3)}'
print(f'Beginning training run {run_id}')
writer = SummaryWriter(f'logs/{run_id}') 
device = torch.device("cuda")
dataroot = "data/celeba"
workers = 16
batch_size = 40
image_size = 256
n_epochs = 5
loss_fn = F.mse_loss
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
opt = torch.optim.AdamW(net.parameters(), lr=4e-4) 
running_loss = 0.0
i = 0
pbar = tqdm(total=len(train_dataloader) * n_epochs)
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        # Get some data and prepare the corrupted version
        x = x.to(device) # Data on the GPU
        amount = torch.linspace(0, 1, x.shape[0]).to(device)
        noisy_x = corrupt(x, amount) # Create our noisy x
        # Get the model prediction
        pred = net(noisy_x)
        # Calculate the loss
        loss = loss_fn(pred, x) # How close is the output to the true 'clean' x?
        # Backprop and update the params:
        opt.zero_grad()
        loss.backward(loss)
        opt.step()
        running_loss += loss.item()
        if i % 250 == 249:    # every 500 mini-batches...
            writer.add_scalar('training loss', running_loss / 500, i)
            img_stack_0 = torch.stack((noisy_x[-1],pred[-1],x[-1]))
            writer.add_images('Noisiest Image Sampls', img_stack_0, epoch * len(train_dataloader) + i)
            img_stack_1 = torch.stack((noisy_x[batch_size//2],pred[batch_size//2],x[batch_size//2]))
            writer.add_images('Middlest Noisy Image Sampls', img_stack_1, i)
        i += 1
        pbar.update(1)
pbar.close()
writer.close()
print(f'Training for model id {run_id} completed')
PATH = f'models/{run_id}.pth'
torch.save(net.state_dict(), PATH)
# net = BasicUNet().to(device)
# net.load_state_dict(torch.load(PATH))
# Sampling
n_steps = 20
x = torch.rand(4, 3, 256, 256).to(device)
q = x
for i in range(n_steps):
    with torch.no_grad():
        pred = net(x)
        q = net(q)
    mix_factor = 1/(n_steps - i)
    x = x*(1-mix_factor) + pred*mix_factor
    for j, img in enumerate(pred): # j = num of fake images
        save_image(img, f'generated_images/{run_id}/celeb_step_{i}_img_{j}.png')
    for j, img in enumerate(q): # j = num of fake images
        save_image(img, f'generated_images/{run_id}/celeb_step_{i}_img_{j}.png')
save_image(q, f'generated_images/{run_id}/q.png')