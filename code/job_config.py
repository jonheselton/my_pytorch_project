import torch, torchvision
from code.convolution_networks import BasicConvolutionUNet
from code.corruption_functions import corrupt_guassian

config = {
    'loss_fn' : torch.nn.SmoothL1Loss(beta=1.0),
    'optimizer' : torch.optim.AdamW,
    'model' : BasicConvolutionUNet(),
    'run_id' : 'faces_1.0.7_with_msfaces',
    'new_model' : False,
    'model_path' : 'models/faces_1.0.6',
    'device' : 'cuda',
    'workers' : 16,
    'batch_size' : 128,
    'image_size' : 128,
    'shuffle' : False,
    'n_epochs' : 1,
    'noise_modifier' : 0.1,
    'data_root' : 'data/ms-celeb-1m/processed',
    'lr' : 1e-3,
    'corruption' : corrupt_guassian,
    'use_subset' : True,
    'subset_start' : 3354000,
    'subset_end' : 3400000
}
