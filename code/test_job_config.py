import torch, torchvision
from code.convolution_networks import BasicConvolutionUNet
from code.corruption_functions import corrupt_guassian

config = {
    'loss_fn' : torch.nn.SmoothL1Loss(beta=1.0),
    'optimizer' : torch.optim.AdamW,
    'net' : BasicConvolutionUNet(),
    'run_id' : 'faces_1.0.5-4',
    'new_model' : False,
    'model_path' : 'models/faces_1.0.5-3',
    'device' : 'cuda',
    'workers' : 16,
    'prefetch' : 4,
    'batch_size' : 128,
    'image_size' : 128,
    'shuffle' : True,
    'n_epochs' : 4,
    'noise_modifier' : 0.4,
    'data_root' : 'data/celebclass/celeba',
    'lr' : 1e-3,
    'corruption' : corrupt_guassian,
    'use_subset' : False,
    'subset_start' : 3000000,
    'subset_end' : 4600000,
    'noise_increase' : 0.05,
    'noise_increase_step' : 100000,
    'notes' : 'Reduced the lr back down to .001'
}
