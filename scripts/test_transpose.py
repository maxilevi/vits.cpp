import torch
import torch.nn as nn

# Original conv1d transpose layer
stride = 8
padding = 0
ks = 16
dilation = 1
conv_transpose1d = nn.ConvTranspose1d(512, 256, kernel_size=ks, stride=stride, padding=padding, bias=False)
print("Weights shape")
print(conv_transpose1d.weight.shape)

def manual_conv_transpose1d(input, weight, bias=None, stride=1, padding=0):
    batch_size, in_channels, in_length = input.shape
    out_channels, _, kernel_size = weight.shape

    # Calculate the output size not including padding
    output_size = (in_length - 1) * stride + dilation * (kernel_size - 1) + 1

    # Expand the input tensor with stride spaces
    expanded_input = torch.zeros((batch_size, in_channels, in_length * stride))
    expanded_input[:, :, ::stride] = input
    print(expanded_input)

    # Flip the weights and transpose to match the input channels
    weight = weight.flip(-1).transpose(0, 1)

    # Perform the convolution
    # The padding for convolution should be set to kernel_size - 1 to get the same size as 'valid' convolution
    res = nn.functional.conv1d(expanded_input, weight, bias, stride=1, padding=kernel_size - 1)

    # Adjust the output size for the padding
    # We need to trim off padding from both sides and also the extra padded values from the last convolution
    start = padding
    end = -(padding + stride - 1) if stride > 1 else None
    res = res[:, :, start:end]

    return res


x = torch.rand(1, 512, 34)
print("Doing y")
y = conv_transpose1d(x.clone())
print(y.shape)
print("Doing y_sim")
y_sim = manual_conv_transpose1d(x, conv_transpose1d.weight, conv_transpose1d.bias, stride=stride, padding=padding)
print(y_sim.shape)

print("y")
for i in y[0, :1]:
    print(i)

print("y_sim")
for i in y_sim[0, :1]:
    print(i)

print(y)
print(y_sim)
print(torch.allclose(y, y_sim, atol=1e-3))