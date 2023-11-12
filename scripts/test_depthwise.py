import torch
import torch.nn as nn
import torch.nn.functional as F

padding = 2
dilation = 2
kernel_size = 16
stride = 8
groups = 192  # This will be the number of input channels for depthwise convolution

# Initialize a Conv1d layer for comparison
conv = nn.Conv1d(in_channels=groups, out_channels=groups, kernel_size=kernel_size,
                   stride=stride, padding=padding, groups=groups, bias=False, dilation=dilation)

def depthwise_conv1d(x):
    # Manual depthwise convolution
    batch_size, channels, width = x.size()

    # Applying conv1d individually for each channel
    outs = []
    for channel in range(channels):
        # Selecting the individual filter and bias for the current channel
        filter = conv.weight[channel, :, :].view(1, 1, kernel_size)
        #bias = conv.bias[channel]
        # Performing the convolution
        #print("Filter size", filter.shape)
        #print("x size", x[:, channel:channel+1, :].shape)
        out = F.conv1d(x[:, channel:channel+1, :], filter, bias=None, stride=stride, padding=padding, dilation=dilation)
        outs.append(out)
    return torch.cat(outs, dim=1)

# Test with random data
# The size of x must match the in_channels expected by the Conv1d module.
x = torch.randn(1, groups, 200)  # Batch size of 1, 'groups' number of channels, width of 200

print(conv.weight.shape)
y1 = conv(x)
y2 = depthwise_conv1d(x)

# Compare the two outputs
print(torch.allclose(y1, y2, atol=1e-6))  # May need to adjust atol depending on precision
