import torchviz
import torch
import time
from transformers import VitsTokenizer
from vits import VitsHifiGan, VitsModel
from torchviz import make_dot

model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-spa")

text = "Hola, cómo estas?"
inputs = tokenizer(text, return_tensors="pt")

output = model(**inputs).waveform
graph = make_dot(output, params=dict(model.named_parameters()))
graph.render('vits_graph', format='png')