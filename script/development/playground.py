
##
import torch
import torch.nn as nn

##  
N, C_in, D, H, W = 32, 1, 11, 23, 255 
x = torch.randn(N, C_in, D, H, W)
x.shape

##
conv = nn.Conv3d(
    in_channels=1, out_channels=16, kernel_size=(3, 3, 3), 
    stride=1, padding=1, dilation=1, 
    groups=1, bias=True, padding_mode='zeros'
)
x = conv(x)
x.shape

##
relu = nn.ReLU()
x = relu(x)
x.shape

##
conv = nn.Conv3d(
    in_channels=16, out_channels=16, kernel_size=(3, 3, 3), 
    stride=1, padding=1, dilation=1, 
    groups=1, bias=True, padding_mode='zeros'
)
x = conv(x)
x.shape

##
relu = nn.ReLU()
x = relu(x)
x.shape

##
x = x.sum(1)
x.shape

