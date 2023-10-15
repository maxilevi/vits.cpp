from transformers import VitsModel, AutoTokenizer
import torch
import json
from surgeon_pytorch import Inspect, get_layers

model = VitsModel.from_pretrained("facebook/mms-tts-spa")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

# Tokenizing input text
text = "hola mundo"
inputs = tokenizer(text, return_tensors="pt")

# Get the list of all layers to find the embedding layer name
layers = get_layers(model)

# Replace 'embedding_layer_name' with the actual name from the printed layers above
embedding_layer_name = 'text_encoder.embed_tokens'

# Wrap model to inspect the specified layer
model_wrapped = Inspect(model, layer={x : x for x in layers})

# Forward pass
with torch.no_grad():
    output, layers = model_wrapped(**inputs)

# Output the inspected embedding layer values
with open('./scripts/expected_shapes.txt', 'w') as f:
    def transform_shape(v):
        return [('X' if x == inputs['input_ids'].shape[1] else str(x)) for x in v.shape]

    shape_dict = {k: transform_shape(v) for k, v in layers.items() if type(v) is torch.Tensor}
    for k, v in shape_dict.items():
        f.write(f"{k}={','.join(v)}\n")

#print(model.text_encoder.embed_tokens.weight.shape)
#print(model.text_encoder.embed_tokens.weight.shape)
