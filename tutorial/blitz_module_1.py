#!python

import torch
import numpy as np

# tensors are similiar to arrays and matrices, reminiscient of linear algebra

# create tensor from data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# create tensor from numpy
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# create tensors from other tensors, they will retain their shape and data type unless otherwise specified
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

# shape is a tuple that describes the dimensions of the tensor
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# more dimensions - can i create one in five dimensions[[]]
shape_5d = (2,4,3,3,7,)
rand_5d_tensor = torch.rand(shape_5d)
print(f'So many dimensions! \n {rand_5d_tensor}')

# Tensor attributes describe their shape and datatype, as well as the device the are stored on (cpu or gpu (cuda)
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

gpu_tensor = torch.rand(4,5, device = 'cuda')
print(f'Device tensor is stored on: {gpu_tensor.device}')

# Lots of operations for tensors, here are a few examples
# Matric multiplication
rand_tensor = torch.rand(5,4, device = 'cuda')
matmul_tensor = gpu_tensor.matmul(rand_tensor)
print(f'rand_tensor @ gpu_tensor = matmul_tensor \n')
print(f'rand_tensor: \n {rand_tensor}')
print(f'gpu_tensor:\n {gpu_tensor}')
print(f'product of matrix multiplication: \n {matmul_tensor}')

# Element-wise multiplication
mult_tensor = gpu_tensor * gpu_tensor
print(f'Element-wise product: \n {mult_tensor}')

# Addition
add_tensor = rand_tensor + 2
print(f'rand_tensor + 2 \n {add_tensor}')

