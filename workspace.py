import torch
from ITTR_pytorch import HPB
import torch.nn as nn

block = HPB(
    dim=512,              # dimension
    dim_head=32,          # dimension per attention head
    heads=8,              # number of attention heads
    # number of top indices to select along height, for the attention pruning
    attn_height_top_k=16,
    # number of top indices to select along width, for the attention pruning
    attn_width_top_k=16,
    attn_dropout=0.,      # attn dropout
    ff_mult=4,            # expansion factor of feedforward
    ff_dropout=0.         # feedforward dropout
)

# fmap = torch.randn(1, 512, 32, 32)

# out = block(fmap)  # (1, 512, 32, 32)
# print(out.shape)


firstConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7,7), stride=2, padding=3)
firstIN = nn.InstanceNorm2d(num_features=3)
gelu = nn.GELU()

x = torch.randn(8, 3, 256, 256)
x = firstConv(x)
x = firstIN(x)
x = gelu(x)

print(x.shape)

secondConv = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=(3,3), stride=2, padding=1)
secondIN = nn.InstanceNorm2d(num_features=256)

x = secondConv(x)
x = secondIN(x)
x = gelu(x)


print(x.shape)

thirdConv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=2, padding=1)
thirdIN = nn.InstanceNorm2d(num_features=512)

x = thirdConv(x)
x = thirdIN(x)
x = gelu(x)


print(x.shape)

x = block(x)

print(x.shape)

upsample = nn.Upsample(scale_factor=2)

x = upsample(x)

print(x.shape)

fourthConv = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3,3), padding=1)
fourthIN = nn.InstanceNorm2d(num_features=256)
gelu = nn.GELU()

x = fourthConv(x)
x = fourthIN(x)
x = gelu(x)

print(x.shape)

x = upsample(x)

print(x.shape)

fifthConv = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=(3,3), padding=1)
fifthIN = nn.InstanceNorm2d(num_features=3)
gelu = nn.GELU()

x = fifthConv(x)
x = fifthIN(x)
x = gelu(x)

print(x.shape)

x = upsample(x)

print(x.shape)


sixthConv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(7,7), padding=3)
tanh = nn.Tanh()

x = sixthConv(x)
x = tanh(x)
print(x.shape)