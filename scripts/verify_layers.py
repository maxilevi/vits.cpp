from transformers import VitsModel, AutoTokenizer
import scipy
import torch

def load_tensor_from_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_dims = int(lines[0].split()[0])
        shape = tuple(map(int, lines[0].split()[1:]))
        flat_data = []
        for line in lines[1:]:
            for value in line.split():
                flat_data.append(float(value))
        return torch.tensor(flat_data).reshape(shape)

def verify_text_encoder(model, input_ids):
    input_padding_mask = torch.ones_like(input_ids).unsqueeze(-1).float()
    text_encoder_output = model.text_encoder(
        input_ids=input_ids,
        padding_mask=input_padding_mask,
    )
    print(text_encoder_output)
    cpp_output = load_tensor_from_file('./debug/text_encoder_output.txt')

    assert torch.allclose(text_encoder_output, cpp_output, atol=1e-3)


def verify_all(text):
    model = VitsModel.from_pretrained("facebook/mms-tts-spa")

    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    print(f"Input ids: {input_ids} for '{text}'")
    verify_text_encoder(model, input_ids)


if __name__ == '__main__':
    verify_all('Hello world!')
