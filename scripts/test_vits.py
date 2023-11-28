import time
from vits import VitsModel
from transformers import VitsTokenizer
import scipy
import torch
import numpy as np
import time

model = VitsModel.from_pretrained("facebook/mms-tts-spa").cpu()
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-spa")

text = """Hola, cómo estas? Hola, cómo estas? Hola, cómo estas? Hola, cómo estas?"""
sentences = text.split("Hola, cómo estas? Hola, cómo estas? Hola, cómo estas? Hola, cómo estas?")
all_audio = torch.tensor([])
#for sentence in sentences:
#    if not sentence:
#        continue
inputs = tokenizer(text, return_tensors="pt")
print(tokenizer)
print(inputs)
t = time.perf_counter()
with torch.no_grad():
    output = model(**inputs).waveform
print("Took {} ms".format((time.perf_counter() - t) * 1000))
all_audio = torch.cat([all_audio, output[0]], dim=0)


print(all_audio.shape, all_audio.dtype, all_audio)
scipy.io.wavfile.write("test.wav", rate=model.config.sampling_rate, data=all_audio.detach().numpy())
print(model.config.sampling_rate)