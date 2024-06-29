import torch
from tqdm import tqdm
from code.utils import load_model
from torchvision.utils import save_image
from code.convolution_networks import BasicConvolutionUNet

def generate_images(net, n_steps: int, n: int, c: int, hw: int, device: str = 'cuda'):
    n_steps = n_steps
    x = torch.rand(n, c, hw, hw).to(device)
    net = net.to(device)
    for i in tqdm(range(n_steps)):
        noise_amount = torch.ones((x.shape[0],)).to(device) * (1 - (i / n_steps))  # Starting high going low
        with torch.no_grad():
            pred = net(x)
            mix_factor = 1 / (n_steps - i)
            x = x * (1 - mix_factor) + pred * mix_factor
        if i % 100 == 0:
            save_image(x, f'generated_images/noise_test1/{i}.png')
            save_image(pred, f'generated_images/noise_test_pred1/{i}.png')
    save_image(x, f'generated_images/noise_test1/test.png')

def main():
    net = BasicConvolutionUNet()
    net = load_model('models/noise_tests', net)
    generate_images(net, 10000, 16, 3, 128)

if __name__ == '__main__':
    main()