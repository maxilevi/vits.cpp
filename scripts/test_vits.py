from transformers import VitsModel, AutoTokenizer
import scipy
import torch

model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

text = "esto es una prueba en espa;ol"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

print(output.shape, output.dtype)
print(output[0])

scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=output[0].detach().numpy())

for layer in model.state_dict().keys():
    print(layer)