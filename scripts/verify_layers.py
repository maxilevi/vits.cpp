from transformers import AutoTokenizer
from vits import VitsModel
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
    #print(text_encoder_output)
    #cpp_output = load_tensor_from_file('./text_encoder_output.txt')

    assert torch.allclose(text_encoder_output, cpp_output, atol=1e-3)


def verify_all(text):
    model = VitsModel.from_pretrained("facebook/mms-tts-spa")

    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
    inputs = tokenizer(text, return_tensors="pt")
    print(f"Input ids: {inputs.input_ids} for '{text}'")

    with torch.no_grad():
        output = model(**inputs).waveform
    print("OUTPUT", output)


if __name__ == '__main__':
    verify_all('Hello world!')
