from transformers import VitsModel, AutoTokenizer
import torch
# Note: `surgeon_pytorch` isn't publicly available/recognized, so you might want to
# replace this part according to your actual use case or library.
from surgeon_pytorch import Inspect, get_layers

model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

# Tokenizing input text
text = "hola mundo"
inputs = tokenizer(text, return_tensors="pt")

# Get the list of all layers to find the embedding layer name
print(get_layers(model))

# Replace 'embedding_layer_name' with the actual name from the printed layers above
embedding_layer_name = 'text_encoder.embed_tokens'

# Wrap model to inspect the specified layer
model_wrapped = Inspect(model, layer={embedding_layer_name: 'token_embeddings', 'text_encoder.encoder.layers.0.attention': 'position_embeddings'})

# Forward pass
with torch.no_grad():
    output, layers = model_wrapped(**inputs)

# Output the inspected embedding layer values
for name, layer in layers.items():
    print(name, layer)

print(model.text_encoder.embed_tokens.weight.shape)
print(model.text_encoder.embed_tokens.weight.shape)
