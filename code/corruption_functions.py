import torch, random

def add_noise_gaussian(image: torch.Tensor, noise_level: float, device: str = 'cuda') -> torch.Tensor:
    """ Add g noise to an image tensor (CHW) """
    noise = torch.randn(image.shape).to(device) * noise_level
    return image + noise

def corrupt_guassian(images: torch.Tensor, noise_modifier: float = 0.0):
    """ Configures corruption levels for a training batch """
    noisy_images = []
    noise_level = random.uniform(0.4 + noise_modifier, 0.8 + noise_modifier)
    for image in images:
        noisy_images.append(add_noise_gaussian(image.clone(), noise_level))
    return torch.stack(noisy_images)


