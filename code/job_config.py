import torch, torchvision
from code.convolution_networks import BasicConvolutionUNet
from code.corruption_functions import corrupt_guassian

config = {
    'loss_fn' : torch.nn.SmoothL1Loss(beta=1.0),
    'optimizer' : torch.optim.AdamW,
    'model' : BasicConvolutionUNet(),
    'run_id' : 'faces_1.0.5-2',
    'new_model' : False,
    'model_path' : 'models/faces_1.0.5-1',
    'device' : 'cuda',
    'workers' : 16,
    'prefetch' : 4,
    'batch_size' : 128,
    'image_size' : 128,
    'shuffle' : False,
    'n_epochs' : 16,
    'noise_modifier' : 0.4,
    'data_root' : 'data/celebclass/celeba',
    'lr' : 1e-3,
    'corruption' : corrupt_guassian,
    'use_subset' : False,
    'subset_start' : 3000000,
    'subset_end' : 4600000,
    'noise_step' : 0.05,
    'noise_step_iteration' : 100000
}
