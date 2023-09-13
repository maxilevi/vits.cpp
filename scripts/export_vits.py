import torch
import struct
from transformers import VitsModel

def serialize_state_dict_to_binary(state_dict, file_name):
    with open(file_name, 'wb') as f:
        for key, tensor in state_dict.items():
            # Write tensor name length and bytes
            tensor_name_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(tensor_name_bytes)))
            f.write(tensor_name_bytes)

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
            f.write(struct.pack('<B', tensor_type))

            # Write tensor shape length and bytes
            tensor_shape_bytes = ' '.join(map(str, tensor.shape)).encode('utf-8')
            f.write(struct.pack('<I', len(tensor_shape_bytes)))
            f.write(tensor_shape_bytes)

            # Write tensor data bytes length and bytes
            tensor_bytes = tensor.numpy().tobytes()
            f.write(struct.pack('<I', len(tensor_bytes)))
            f.write(tensor_bytes)

if __name__ == '__main__':
    model_name = "facebook/mms-tts-spa"
    model = VitsModel.from_pretrained(model_name)
    serialize_state_dict_to_binary(model.state_dict(), f'vits-{model_name.replace("/", "-")}.ggml')
    print("Done!")