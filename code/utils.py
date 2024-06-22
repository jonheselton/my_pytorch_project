import random, string, os, torch, torchvision, time, tqdm, os
from torch import nn
from torch.utils.tensorboard import SummaryWriter

def load_model(path, model):
    model.load_state_dict(torch.load(path))
    return model

def save_model(path, model):
    torch.save(model.state_dict(), path)

def save_images_to_tb(tag: str, tensor_stack: torch.Tensor, writer: SummaryWriter, n: int):
    """Save images (as NCHW tensor) to tensorboard"""
    img_grid = torchvision.utils.make_grid(img_stack)
    # write to tensorboard
    writer.add_image(tag, img_grid, n)

def initialize_weights_kaiming(model, activation):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity=activation)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
