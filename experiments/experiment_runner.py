import random, string, os, torch, torchvision, time, tqdm, os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
options = {}
def load_model(path, model):
    pass
def save_model(path, model):
    pass

def training_run(run_name, model, epoch, dataset, loss_fn, optimizer, corruption, transforms):
    pass

