import torch
import torch.nn as nn
conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
conv2 = nn.Conv2d(1, 2, kernel_size=3)

inputs = torch.Tensor([[[[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]]])
print("input size: ", inputs.shape)
outputs1 = conv1(inputs)
print("output1 size: ", outputs1.shape)
outputs2 = conv2(inputs)
print("output2 size: ", outputs2.shape)