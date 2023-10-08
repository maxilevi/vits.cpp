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

if __name__ == '__main__':
    model_name = "facebook/mms-tts-spa"
    model = VitsModel.from_pretrained(model_name)
    print(model.config)
    serialize_model_to_binary(model.config, model.state_dict(), f'./scripts/vits-spanish.ggml')
    print("Done!")
