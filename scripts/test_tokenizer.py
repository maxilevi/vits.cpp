from transformers import VitsModel, AutoTokenizer
import scipy
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

text = "esto es una prueba en español. gracias"
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs['input_ids']
print(input_ids.shape, input_ids)

inputs = tokenizer("esto es una prueba", return_tensors="pt")
input_ids = inputs['input_ids']
print(input_ids.shape, input_ids)