from transformers import VitsModel, AutoTokenizer
import scipy
import torch

def verify_text_encoder(model, input_ids):
    text_encoder_output = model.text_encoder(
        input_ids=input_ids
    )
    print(text_encoder_output)


def verify_all(text):
    model = VitsModel.from_pretrained("facebook/mms-tts-spa")

    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    verify_text_encoder(model, input_ids)


if __name__ == '__main__':
    verify_all('Hello world!')
