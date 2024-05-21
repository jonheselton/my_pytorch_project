import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

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
writer = SummaryWriter("logs/training_run") 
device = torch.device("cuda")
dataroot = "data/celeba"
workers = 16
batch_size = 32
image_size = 256
# Use celeb dataloader instead
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# create smaller subsets of the total data
total_images = len(dataset)
num_subsets = 9
images_per_subset = int(total_images / num_subsets)
all_subsets = []
# Randomize the order
all_indices = np.random.permutation(total_images)
start_idx = 0
for i in range(num_subsets):
  subset_indices = all_indices[start_idx:start_idx + images_per_subset]
  # Create the subset
  current_subset = Subset(dataset, subset_indices)
  all_subsets.append(current_subset)
  start_idx += images_per_subset
# Create the dataloaders
# Dataloader contains a data tensor along with a tensor of classes{?comfirm?}
# Dataloader length = batch size / data objects
data_loaders = []
for i in all_subsets:
    subset_dataloader = DataLoader(i, batch_size=batch_size, shuffle=True, num_workers=workers)
    data_loaders.append(subset_dataloader)
# How many runs through the data should we do?
n_epochs = 1
# Create the network
net = BasicUNet()
net.to(device)
# Our loss function
loss_fn = nn.SmoothL1Loss()
# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 
# Keeping a record of the losses for later viewing
data_set_number = 0
# The training loops
small_data_loaders = data_loaders[2:3]
for train_dataloader in small_data_loaders:
    losses_subset = []
    for epoch in range(n_epochs):
        losses_epoch = []
        i = 0
        for x, y in train_dataloader:
            i += 1
            # Get some data and prepare the corrupted version
            x = x.to(device) # Data on the GPU
            noisy_x = corrupt(x, 0) # Create our noisy x
            # Get the model prediction
            pred = net(noisy_x)
            # Calculate the loss
            loss = loss_fn(pred, x) # How close is the output to the true 'clean' x?
            writer.add_scalar(f"Loss/train {len(losses_subset)}", loss.item(), epoch)
            # Backprop and update the params:
            opt.zero_grad()
            loss.backward()
            opt.step()
            # Store the loss for later
            losses_epoch.append(loss.item())
            losses_subset.append(loss.item())
            
            print(opt.state_dict().keys())
            writer.add_scalar("Gradients/Layer1_weight_update", opt.state_dict()["model.layer1.weight"].norm(), len(losses_epoch), len(losses_subset))# Track weight updates of a specific layer
            
            writer.add_scalar("Gradients/Layer2_weight_update", opt.state_dict()["model.layer2.weight"].norm(), len(losses_epoch), len(losses_subset))
            writer.add_scalar("Gradients/Layer3_weight_update", opt.state_dict()["model.layer3.weight"].norm(), len(losses_epoch), len(losses_subset))
            writer.add_image(f"Input_Image {len(subset_losses)}", x[0], epoch)  # Log the first image in the batch
            writer.add_image(f"Predicted_Mask {len(subset_losses)}", pred[0], epoch)  # Log the first predicted mask in the batch

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss: %.4f'
                % (epoch + 1, n_epochs, i, len(train_dataloader),
                    loss.item()))
        # Print our the average of the loss values for this epoch:
        avg_loss_epoch = sum(losses_epoch)/len(losses_epoch)
        print(f'Finished dataset {data_set_number + 1}, epoch {epoch + 1}. Average loss for this epoch: {avg_loss_epoch:05f}')
    avg_loss_subset = sum(losses_subset)/len(losses_subset)
    print(f'Finished dataset {data_set_number + 1}, Average loss for this subset {avg_loss_subset:05f}')
    data_set_number += 1
writer.flush()
# Save the model
PATH = './models/tensor_board_new_corruption_subsets_unet_diffusion.pth'
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
        save_image(img, f'generated_images/relu_celeb_step_{i}_img_{j}.png')
save_image(q, 'generated_images/q.png')