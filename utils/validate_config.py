import torch, torchvision
from code.convolution_networks import BasicConvolutionUNet
from code.corruption_functions import corrupt_guassian

class TrainingConfig():
    def self.__init__(
            loss_fn: str,
            optimizer: str,
            net: str,
            run_id: str,
            new_model: bool,
            model_path: str,
            device: str,
            workers: int,
            prefetch: int,
            batch_size: int,
            image_size: int,
            shuffle: bool,
            n_epochs: int,
            noise_modifier: float,
            data_root: str,
            lr: float,
            corruption: str,
            use_subset: bool,
            subset_start: int,
            subset_end: int,
            noise_increase: float,
            no8ise_increase_step: int
        ) -> TrainingConfig:
        self.loss_fn = loss_functions[loss_fn.lower()]['fn'](**loss_functions[loss_fn.lower()]['defaults']),
        self.net = neural_networks[net.lower]()
        self.lr = lr,
        self.optimizer = optimizers[optimizer.lower](lr = self.lr),
        self.run_id = run_id,
        self.new_model = new_model,
        self.model_path = model_path,
        self.device = device,
        self.workers = workers,
        self.prefetch = prefetch,
        self.batch_size = batch_size,
        self.image_size = image_size,
        self.shuffle = shuffle,
        self.n_epochs = n_epochs,
        self.noise_modifier = noise_modifier,
        self.data_root = data_root,
        self.corruption = corruption,
        self.use_subset = use_subset,
        self.subset_start = subset_start,
        self.subset_end = subset_end,
        self.noise_increase = noise_increase,
        self.noise_increase_step = noise_increase_step


loss_functions = {
    'smoothl1loss' : {
        'fn' : torch.nn.SmoothL1Loss,
        'defaults' : {
            'beta' : 1
        } 
    }
}
neural_networks = {
    'basicconfolutionunet' : BasicConvolutionUNet
}

optimizers = {
    'adamw' : torch.optim.AdamW,
}

corruption_functions = {
    'corrupt_guassian' = corrupt_guassian
}