import random, string, torch, torchvision
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# """ Tensorboard Projection demo """
# # transform = transforms.Compose(
# #     [transforms.ToTensor(),
# #      transforms.CenterCrop(128),
# #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# def randomword(length):
#    letters = string.ascii_lowercase
#    return ''.join(random.choice(letters) for i in range(length))
# rand_string = randomword(4)
# # writer = SummaryWriter(f'logs/scratch_{rand_string}') 
# # dataroot = "data/celeba"
# # image_size = 256
# # n = 100
# # # Use celeb dataloader instead
# # dataset = dset.ImageFolder(root=dataroot, transform=transform)
# # # Create a tuple containing a bunch of images to stack
# # img_list = []
# # for i in range(n):
# #     img_list += dataset[i][0].unsqueeze(0)
# # img_tuple = tuple(img_list)
# # img_stack = torch.stack(img_tuple)
# # feat = img_stack.reshape(n, -1)
# # writer.add_embedding(feat, label_img=img_stack)
# # writer.close()

""" Tensorboard image demo """
# for i in range(16):
#     img_list += dataset[i][0].unsqueeze(0)
# img_tuple = tuple(img_list)
# img_stack = torch.stack(img_tuple)
# writer = SummaryWriter(f'logs/scratch_{rand_string}') 
# # create grid of images
# img_grid = torchvision.utils.make_grid(img_stack)
# # write to tensorboard
# writer.add_image(f'writing images to tensorboard demo', img_grid)
# # # writer.close()

# """ Tensorboard Model training demo """
# def corrupt(x, amount):
#   """Corrupt the input `x` by mixing it with noise according to `amount`"""
#   noise = torch.rand_like(x)
#   amount = amount.view(-1, 1, 1, 1) # Sort shape so broadcasting works
#   return x*(1-amount) + noise*amount 

# class BasicUNet(nn.Module):
#     """A minimal UNet implementation."""
#     def __init__(self, in_channels=3, out_channels=3):
#         super().__init__()
#         self.down_layers = torch.nn.ModuleList([ 
#             nn.Conv2d(in_channels, 128, kernel_size=5, padding=2),
#             nn.Conv2d(128, 256, kernel_size=5, padding=2),
#             nn.Conv2d(256, 256, kernel_size=5, padding=2),
#         ])
#         self.up_layers = torch.nn.ModuleList([
#             nn.Conv2d(256, 256, kernel_size=5, padding=2),
#             nn.Conv2d(256, 128, kernel_size=5, padding=2),
#             nn.Conv2d(128, out_channels, kernel_size=5, padding=2), 
#         ])
#         self.act = nn.LeakyReLU() # The activation function
#         self.downscale = nn.MaxPool2d(2)
#         self.upscale = nn.Upsample(scale_factor=2)
#     def forward(self, x):
#         h = []
#         for i, l in enumerate(self.down_layers):
#             x = self.act(l(x)) # Through the layer and the activation function
#             if i < 2: # For all but the third (final) down layer:
#               h.append(x) # Storing output for skip connection
#               x = self.downscale(x) # Downscale ready for the next layer
#         for i, l in enumerate(self.up_layers):
#             if i > 0: # For all except the first up layer
#               x = self.upscale(x) # Upscale
#               x += h.pop() # Fetching stored output (skip connection)
#             x = self.act(l(x)) # Through the layer and the activation function
#         return x

# dataroot = "data/celeba"
# workers = 16
# batch_size = 64
# image_size = 128
# dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
#                                transforms.CenterCrop(image_size),
#                                transforms.ToTensor(),
#                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                            ]))
# # create smaller subsets of the total data
# total_images = len(dataset)
# num_subsets = 9
# images_per_subset = int(total_images / num_subsets)
# all_subsets = []
# # Randomize the order
# all_indices = np.random.permutation(total_images)
# start_idx = 0
# for i in range(num_subsets):
#   subset_indices = all_indices[start_idx:start_idx + images_per_subset]
#   # Create the subset
#   current_subset = Subset(dataset, subset_indices)
#   all_subsets.append(current_subset)
#   start_idx += images_per_subset
# data_loaders = []
# for i in all_subsets:
#     subset_dataloader = DataLoader(i, batch_size=batch_size, shuffle=True, num_workers=workers)
#     data_loaders.append(subset_dataloader)
# n_epochs = 1
# # Create the network
# if torch.cuda.is_available():
#     device = 'cuda'
# else:
#     device = 'cpu'
# net = BasicUNet()
# net.to(device)
# opt = torch.optim.AdamW(net.parameters(), lr=4e-4)
# loss_fn = F.mse_loss

# # Keeping a record of the losses for later viewing
# data_set_number = 1
# # The training loops
# small_data_loader = data_loaders[0]
# running_loss = 0.0
# writer = SummaryWriter(f'logs/training_{rand_string}') 
# for epoch in range(n_epochs):
#     i = 0
#     for x, y in small_data_loader:
#         i += len(x)
#         # Get some data and prepare the corrupted version
#         x = x.to(device) # Data on the GPU
#         amount = torch.linspace(0, 1, x.shape[0]).to(device)
#         noisy_x = corrupt(x, amount) # Create our noisy x
#         # Get the model prediction
#         pred = net(noisy_x)
#         # Calculate the loss
#         loss = loss_fn(pred, x) # How close is the output to the true 'clean' x?
#         # Backprop and update the params:
#         opt.zero_grad()
#         loss.backward(loss)
#         opt.step()
#         running_loss += loss.item()
#         if i % 100 == 99: # every 100 mini-batches...
#             writer.add_scalar('training loss',
#                         running_loss / 100,
#                         epoch * len(small_data_loader) + i)
#             target_img_grid = torchvision.utils.make_grid(x[0:16])
#             noisy_img_grid = torchvision.utils.make_grid(noisy_x[0:16])
#             pred_img_grid = torchvision.utils.make_grid(pred[0:16])
#             writer.add_image(f'Sample input',target_img_grid, epoch * len(small_data_loader) + i)
#             writer.add_image(f'Sample noisy',noisy_img_grid, epoch * len(small_data_loader) + i)
#             writer.add_image(f'Sample predictions',pred_img_grid, epoch * len(small_data_loader) + i)
#             running_loss = 0.0

# writer.close()
# PATH = './models/tensor_board_new_corruption_subsets_unet_diffusion.pth'
# torch.save(net.state_dict(), PATH)

""" Tensorboard Inspect a model with a graph """

class BUNet(nn.Module):
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
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.CenterCrop(128),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataroot = "data/celeba"
image_size = 256
n = 100
# Use celeb dataloader instead
dataset = dset.ImageFolder(root=dataroot, transform=transform)
# Create a tuple containing a bunch of images to stack
img_list = []
# Get some images from above
for i in range(16):
    img_list += dataset[i][0].unsqueeze(0)
img_tuple = tuple(img_list)
img_stack = torch.stack(img_tuple).to('cuda')

bunet_final = BUNet().to('cuda')
PATH_1 = 'models/diffusion_hf_01_jpvok-final.pth'
bunet_final.load_state_dict(torch.load(PATH_1))
PATH_2 = 'models/diffusion_hf_01_jpvok_1.pth'
bunet_epoch1 = BUNet().to('cuda')
bunet_epoch1.load_state_dict(torch.load(PATH_2))

writer1 = SummaryWriter('logs/tb_model_graph_epoch1/')
writer1.add_graph(bunet_epoch1, img_stack)
writer2 = SummaryWriter('logs/tb_model_graph_final/')
writer2.add_graph(bunet_final, img_stack)
writer1.close()
writer2.close()
