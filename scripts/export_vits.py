import torch
import struct
from transformers import VitsModel

def serialize_model_to_binary(config, state_dict, file_name):
    with open(file_name, 'wb') as f:
        # Write config
        items = config.to_diff_dict().items()
        f.write(struct.pack('<I', len(items)))
        for key, value in items:
            key_bytes = key.encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack('<I', len(value_bytes)))
            f.write(value_bytes)

        # Write state dict
        tensors = state_dict.items()
        f.write(struct.pack('<I', len(tensors)))
        for key, tensor in tensors:
            # Write tensor name length and bytes
            tensor_name_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(tensor_name_bytes)))
            f.write(tensor_name_bytes)
            print(len(tensor_name_bytes), tensor_name_bytes)

            # Write tensor type
            type_mapping = {
                torch.float32: 0,
                torch.float16: 1,
                torch.int64: 2,
                torch.int32: 3,
                torch.int8: 4
            }
            tensor_type = type_mapping.get(tensor.dtype, None)
            if tensor_type is None:
                raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")
            f.write(struct.pack('<I', tensor_type))

            # Write tensor shape length (number of dimensions) and shape values
            tensor_rank = len(tensor.shape)
            f.write(struct.pack('<I', tensor_rank))
            for dim in tensor.shape[::-1]:
                f.write(struct.pack('<I', dim))

            # Write tensor data bytes length and bytes
            tensor_bytes = tensor.numpy().tobytes()
            f.write(struct.pack('<I', len(tensor_bytes)))
            f.write(tensor_bytes)

def merge_weight_normalization(state_dict):
    v_keys = []
    g_keys = []

    for key in state_dict.keys():
        if "parametrizations.weight.original" in key:
            if key.endswith("0"):
                g_keys.append(key)
            if key.endswith("1"):
                v_keys.append(key)

    # Compute normalized weights and update state_dict
    for v_key, g_key in zip(v_keys, g_keys):
        print(v_key, g_key)
        w = state_dict[g_key] * (state_dict[v_key] / torch.norm(state_dict[v_key]))

        # Replace the "original" weight key in state_dict with normalized weights
        original_key = g_key.replace(".parametrizations.weight.original0", ".weight")
        state_dict[original_key] = w
        print(original_key, w.shape)
        # Optionally, delete the g and v keys from state_dict if you won't need them anymore
        del state_dict[g_key]
        del state_dict[v_key]

    return state_dict

if __name__ == '__main__':
    model_name = "facebook/mms-tts-spa"
    model = VitsModel.from_pretrained(model_name)
    print(model.config)
    serialize_model_to_binary(model.config, merge_weight_normalization(model.state_dict()), f'./scripts/vits-spanish.ggml')
    print("Done!")
