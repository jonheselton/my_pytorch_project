import torch, torchvision, time, tqdm
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
from code.convolution_networks import BasicConvolutionUNet
from code.corruption_functions import corrupt_guassian
from code.job_config import config
from code.utils import *

def training_run(
                loss_fn, 
                optimizer, 
                model, 
                run_id, 
                model_path, 
                device, 
                workers,
                batch_size,
                image_size,
                shuffle,
                n_epochs,
                noise_modifier,
                data_root,
                lr,
                corruption,
                subset_start,
                subset_end,
                use_subset = False,
                new_model = False):
    fn_start = time.time()
    model.to(device)
    if load_model:
        load_model(model_path, model)
    if new_model:
        initialize_weights_kaiming(model)
    opt = optimizer(model.parameters(), lr=lr)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomGrayscale(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Resize(image_size),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = dset.ImageFolder(root=data_root, transform=transforms)
    if use_subset:
        dataset = torch.utils.data.Subset(dataset, range(subset_start, subset_end))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, prefetch_factor=4)
    writer = SummaryWriter(f'logs/{run_id}')
    writer.add_text('model', f'{type(model)}')
    writer.add_text('Loaded model', f'{model_path}')
    writer.add_text('Loss function', f'{loss_fn}')
    writer.add_text('Optimizer', f'{optimizer}')
    writer.add_text('Noise modifier',f'{noise_modifier}')
    writer.add_text('Learning rate', f'{lr}')
    writer.add_text('Corruption function', f'{corruption}')
    writer.add_text('Epochs', f'{n_epochs}')
    writer.add_text('Batch size', f'{batch_size}')
    writer.add_text('Image size', f'{image_size}')
    writer.add_text('workers', f'{workers}')
    writer.add_text('Data', f'{data_root}')
    writer.add_text('Number of images', f'{len(data)}')
    running_loss = 0.0
    i = 0
    print(f'Beggining training run {run_id}\n Preperation took {time.time() - fn_start}')
    pbar = tqdm.tqdm(total=len(data) * n_epochs)
    for epoch in range(n_epochs):
        for x, y in data:
            x = x.to(device)
            noise_modifier = noise_modifier + (0.05 * i//5000)
            noisy_x = corruption(x, noise_modifier) # Create our noisy x
            pred = model(noisy_x)
            loss = loss_fn(pred, x)
            opt.zero_grad()
            loss.backward()
            # writer.add_scalar('noise', noise_level, i)
            # for name, param in model.named_parameters():
            #     writer.add_histogram(f"weights/{name}", param.data, i)
            #     writer.add_histogram(f"gradients/{name}", param.grad.data, i)
            opt.step()
            running_loss += loss.item()
            if i % 250 == 249: 
                writer.add_scalar('training loss', running_loss / 250, i)
                running_loss = 0.0
            if i % 5000 == 4999:
                img_stack_0 = torch.stack((x[-1], noisy_x[-1], pred[-1]))
                writer.add_images('Image Sampls', img_stack_0, i)
            i += 1
            pbar.update(1)
    save_model(f'models/{run_id}', model)
    pbar.close()
    writer.close()
    print(f'Training complete, total duration: {time.time() - fn_start}, total imagets{len(data)}')

def main():
    training_run(**config)

if __name__ == '__main__':
    main()
