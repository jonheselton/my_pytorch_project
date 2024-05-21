import random, string, torch, torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # Convolution -> relu ->
        x = self.pool(F.relu(self.conv1(x)))
        # convolution -> relu ->
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # Linear -> relu ->
        x = F.relu(self.fc1(x))
        # Linear -> relu
        x = F.relu(self.fc2(x))
        # Linear ->
        x = self.fc3(x)
        return x

    """ Tensorboard Projection demo """
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.CenterCrop(128),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def randomword(length):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(length))
rand_string = randomword(4)
writer = SummaryWriter(f'logs/scratch_{rand_string}') 
dataroot = "data/celeba"
image_size = 256
n = 100
# Use celeb dataloader instead
dataset = dset.ImageFolder(root=dataroot, transform=transform)
# Create a tuple containing a bunch of images to stack
img_list = []
for i in range(n):
    img_list += dataset[i][0].unsqueeze(0)
img_tuple = tuple(img_list)
img_stack = torch.stack(img_tuple)
feat = img_stack.reshape(n, -1)
writer.add_embedding(feat, label_img=img_stack)
writer.close()

""" Tensorboard Model graph and image demo """
net = Net()
for i in range(16):
    img_list += dataset[i][0].unsqueeze(0)
img_tuple = tuple(img_list)
img_stack = torch.stack(img_tuple)
writer = SummaryWriter(f'logs/scratch_{rand_string}') 
# create grid of images
img_grid = torchvision.utils.make_grid(img_stack)
# write to tensorboard
writer.add_image(f'writing images to tensorboard demo', img_grid)
writer.close()


# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# for epoch in range(2):

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + loss + backward + optimize (adjust params/weights)
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
#             running_loss = 0.0

# print('Finished Training')
