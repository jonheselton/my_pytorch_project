#!python

import torch
if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU is NOT available')

# Print the GPU name
print(torch.cuda.get_device_name())

# Assign a tensor to the GPU
tensor_rand = torch.rand(3, 4, device = 'cuda')

print(tensor_rand)