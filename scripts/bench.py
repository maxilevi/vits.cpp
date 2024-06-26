from transformers import VitsTokenizer, VitsModel
import torch
import time

model = VitsModel.from_pretrained("facebook/mms-tts-spa").cpu()
tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-spa")

text = "Cada amanecer trae consigo nuevas oportunidades para crecer y aprender"
t = time.perf_counter()
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    for _ in range(1):
        output = model(**inputs).waveform
print("Took {} ms".format(((time.perf_counter() - t) * 1000) / 100))