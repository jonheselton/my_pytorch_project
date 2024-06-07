import random, string, torch, torchvision
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def projection(writer, data, n):
    img_list = []
    for i in range(n):
        img_list += data[i][0].unsqueeze(0)
    img_tuple = tuple(img_list)
    img_stack = torch.stack(img_tuple)
    feat = img_stack.reshape(n, -1)
    writer.add_embedding(feat, label_img=img_stack)
    writer.close()

def model_graph(writer, model, data):
    img_list = []
    for i in range(24):
        img_list += data[i][0].unsqueeze(0)
    img_tuple = tuple(img_list)
    img_stack = torch.stack(img_tuple)
    writer.add_graph(model, img_stack)

def main():
    pass

if __name__ == '__main__':
    main()

