import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.utils.parametrizations as parametrizations

def create_conv1d_with_weight_norm(speaker_embedding_size, hidden_size, num_layers):
    """
    Create a Conv1D layer with weight normalization.
    """
    conv1d = nn.Conv1d(speaker_embedding_size, 2 * hidden_size * num_layers, 1)
    return parametrizations.weight_norm(conv1d, name='weight')

def remove_weight_norm_and_merge_weights(conv1d_with_wn):
    """
    Remove weight normalization from a Conv1D layer and merge weights.
    """
    # Remove weight norm
    conv1d = parametrize.remove_parametrizations(conv1d_with_wn, 'weight')

    # Get the merged weight and bias
    merged_weight = conv1d.weight.data
    merged_bias = conv1d.bias.data if conv1d.bias is not None else None

    # Create a new Conv1D layer with the merged weights and bias
    new_conv1d = nn.Conv1d(conv1d.in_channels, conv1d.out_channels, conv1d.kernel_size)
    new_conv1d.weight.data = merged_weight
    new_conv1d.bias.data = merged_bias if merged_bias is not None else torch.zeros(new_conv1d.out_channels)

    return new_conv1d

def test_conv1d_layers(conv1d_wn, conv1d_merged, input_tensor):
    """
    Test if the output of two Conv1D layers (with and without weight norm) are the same.
    """
    output_wn = conv1d_wn(input_tensor)
    output_merged = conv1d_merged(input_tensor)
    print(conv1d_merged)
    print(conv1d_wn)

    return torch.allclose(output_wn, output_merged)

# Configurations
speaker_embedding_size = 128  # Example size
hidden_size = 256             # Example size
num_layers = 4                # Example number of layers

# Create a Conv1D layer with weight normalization
conv1d_with_wn = create_conv1d_with_weight_norm(speaker_embedding_size, hidden_size, num_layers)

# Create a new Conv1D layer by removing weight norm and merging weights
conv1d_merged = remove_weight_norm_and_merge_weights(conv1d_with_wn)

# Create a random input tensor
input_tensor = torch.randn(1, speaker_embedding_size, 10)  # Example input tensor

# Test the layers
test_result = test_conv1d_layers(conv1d_with_wn, conv1d_merged, input_tensor)
print(test_result)
