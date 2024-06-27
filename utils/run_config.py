import torch, torchvision, tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
from code.utils import load_model, save_model
from code.convolution_networks import BasicConvolutionUNet
from code.corruption_functions import corrupt_guassian
from dataclasses import dataclass

config = {
    'loss_fn' : 'SmoothL1Loss',
    'optimizer' : 'AdamW',
    'net' : 'BasicConvolutionUNet',
    'run_id' : 'faces_1.1.01_ms_faces',
    'new_model' : False,
    'model_path' : 'models/faces_1.1_ms_faces',
    'device' : 'cuda',
    'workers' : 16,
    'prefetch' : 4,
    'batch_size' : 128,
    'image_size' : 128,
    'shuffle' : True,
    'n_epochs' : 1,
    'noise_modifier' : 0.4,
    'data_root' : 'data/ms-celeb-1m/processed',
    'lr' : 2e-4,
    'corruption' : 'corrupt_guassian',
    'use_subset' : True,
    'subset_start' : 1000000,
    'subset_end' : 2000000,
    'noise_increase' : 0.05,
    'noise_increase_step' : 10000,
    'notes' : 'Continuing to run through MS-celeb dataset'
}

@dataclass
class TrainingConfig:
    device: str
    loss_fn: str
    net: str
    lr: float
    optimizer: str
    run_id: str
    new_model: bool
    model_path: str
    workers: int
    prefetch: bool
    batch_size: int
    image_size: int
    shuffle: bool
    n_epochs: int
    noise_modifier: float
    data_root: str
    corruption: str
    use_subset: bool
    subset_start: int
    subset_end: int
    noise_increase: float
    noise_increase_step: int
    notes: str

    def run(self):
        loss_fn = loss_functions[self.loss_fn.lower()]['fn'](beta = 1)
        net = neural_networks[self.net.lower()]()
        net.to(self.device)
        if self.model_path:
            load_model(self.model_path, net)
        if self.new_model:
            initialize_weights_kaiming(net)
        corruption = corruption_functions[self.corruption.lower()]
        opt = optimizers[self.optimizer.lower()]
        opt = opt(net.parameters(), lr=self.lr)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomGrayscale(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(self.image_size),
            torchvision.transforms.CenterCrop(self.image_size),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = dset.ImageFolder(root=self.data_root, transform=transforms)
        if self.use_subset:
            dataset = torch.utils.data.Subset(dataset, range(self.subset_start, self.subset_end))
        data = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.workers, prefetch_factor=self.prefetch)
        writer = SummaryWriter(f'logs/{self.run_id}')
        writer.add_text('config', f'#Run: {self.run_id} \n --- \n NeuralNet: {type(net).__name__} - Loaded model:{self.model_path} \n --- \n Loss function: {self.loss_fn} \n Corruption function: {self.corruption} - Noise modifier: {self.noise_modifier} \n --- \n Optimizer {self.optimizer} - Learning rate:{self.lr} \n --- \n Epochs: {self.n_epochs}  -  Batch size:  {self.batch_size}  -  Number of workers: {self.workers}  ---  Data: {self.data_root}  -  Number of images: {len(dataset)}  - Number of steps per epoch {len(data)} - Image size: {self.image_size}  \n --- \n {self.notes}')
        running_loss = 0.0
        i = 0
        print(f'Beggining training run {self.run_id}')
        pbar = tqdm.tqdm(total=len(data) * self.n_epochs)
        for epoch in range(self.n_epochs):
            for x, y in data:
                x = x.to(self.device)
                noise_modifier = self.noise_modifier + (self.noise_increase * i//self.noise_increase_step)
                noisy_x = corruption(x, noise_modifier)
                pred = net(noisy_x)
                loss = loss_fn(pred, x)
                opt.zero_grad()
                loss.backward()
                # writer.add_scalar('noise', noise_level, i)
                for name, param in net.named_parameters():
                    writer.add_histogram(f"weights/{name}", param.data, i)
                    writer.add_histogram(f"gradients/{name}", param.grad.data, i)
                opt.step()
                running_loss += loss.item()
                if i % 250 == 249: 
                    writer.add_scalar('training loss', running_loss / 250, i)
                    running_loss = 0.0
                if i % 1000 == 999:
                    img_stack_0 = torch.stack((x[-1], noisy_x[-1], pred[-1]))
                    writer.add_images('Image Sampls', img_stack_0, i)
                i += 1
                pbar.update(1)
        save_model(f'models/{self.run_id}', net)
        pbar.close()
        writer.close()
        print(f'Training complete')


    

loss_functions = {
    'smoothl1loss' : {
        'fn' : torch.nn.SmoothL1Loss,
        'defaults' : {
            'beta' : 1
        } 
    }
}
neural_networks = {
    'basicconvolutionunet' : BasicConvolutionUNet,
}

optimizers = {
    'adamw' : torch.optim.AdamW,
}

corruption_functions = {
    'corrupt_guassian' : corrupt_guassian,
}

def main():
    run = TrainingConfig(**config)
    run.run()

if __name__ == '__main__':
    main()
