# Scratch file for tensorboard stuff
import random, string
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

def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))

rand_string = randomword(4)
writer = SummaryWriter(f'logs/scratch_{rand_string}') 
dataroot = "data/celeba"
image_size = 256
n = 100
# Use celeb dataloader instead
dataset = dset.ImageFolder(root=dataroot, transform=transform, train=False)
# Create a tuple containing a bunch of images to stack
img_list = []
for i in range(n):
    img_list += dataset[i][0].unsqueeze(0)
img_tuple = tuple(img_list)
img_stack = torch.stack(img_tuple)
feat = img_stack(n, -1)
writer.add_embedding(feat, label_img=img_stack)
writer.close()