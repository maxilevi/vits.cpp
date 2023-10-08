from transformers import AutoTokenizer
from vits import VitsModel
import scipy
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

text = "hola mundo"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform