import torch
import struct
from transformers import VitsModel, VitsTokenizer

def serialize_model_to_binary(config, state_dict, tokenizer, file_name):
    with open(file_name, 'wb') as f:
        # Write tokenizer
        assert not tokenizer.phonemize
        assert not tokenizer.is_uroman
        # Write tokenizer vocab
        vocab = tokenizer.get_vocab()
        f.write(struct.pack('<I', len(vocab)))
        for key, value in vocab.items():
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack('<I', value))
        # Write tokenizer options
        f.write(struct.pack('<I', tokenizer.add_blank))
        f.write(struct.pack('<I', tokenizer.normalize))
        # Write pad token and unk token
        pad_token_bytes = tokenizer.pad_token.encode('utf-8')
        f.write(struct.pack('<I', len(pad_token_bytes)))
        f.write(pad_token_bytes)
        unk_token_bytes = tokenizer.unk_token.encode('utf-8')
        f.write(struct.pack('<I', len(unk_token_bytes)))
        f.write(unk_token_bytes)

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
            assert not 'original0' in key
            # Write tensor name length and bytes
            tensor_name_bytes = key.encode('utf-8')
            f.write(struct.pack('<I', len(tensor_name_bytes)))
            f.write(tensor_name_bytes)
            #print(len(tensor_name_bytes), tensor_name_bytes)

            # Write tensor type
            type_mapping = {
                torch.float32: 0,
                torch.float16: 1
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

def remove_weight_norm_and_convert_to_fp16(module, full_name=''):
    import torch
    import torch.nn.utils.parametrize as parametrize

    for name, submodule in module.named_children():
        # Check if the submodule is an instance of Conv1d or ConvTranspose1d
        is_parametrized = str(type(submodule)) == "<class 'torch.nn.utils.parametrize.ParametrizedConv1d'>"
        if isinstance(submodule, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)) or is_parametrized:
            # Convert weights to float16
            if is_parametrized:
                parametrize.remove_parametrizations(submodule, 'weight', leave_parametrized=True)
                # Optionally print a message
                print(f"Removed weight norm")

            #if not 'resblocks' in full_name:
            #submodule.weight.data = submodule.weight.data.to(torch.float16)
            #print(f"Converted {name} weights to float16")

        # Recursively apply to children modules
        remove_weight_norm_and_convert_to_fp16(submodule, full_name + '.' + name)

    return module

if __name__ == '__main__':
    for model_name, file_name in [("facebook/mms-tts-eng", f'./scripts/vits-english.ggml'), ("facebook/mms-tts-spa", f'./scripts/vits-spanish.ggml')]:
        model = VitsModel.from_pretrained(model_name)
        tokenizer = VitsTokenizer.from_pretrained(model_name)
        model = remove_weight_norm_and_convert_to_fp16(model)
        serialize_model_to_binary(model.config, model.state_dict(), tokenizer, file_name)
        print(f"Done! Exported {model_name} to {file_name}")
