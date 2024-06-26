# vits-cpp

a cpp ggml port of "VITS: Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech." for use in mobile devices. 

# How to use

Clone and fetch all submodules
```
git@github.com:maxilevi/vits.cpp.git
cd vits.cpp
git submodule update --init --rec   ursive
```

Fetch the models
```
git lfs pull
```

Alternatively you can export the models yourself from huggingface 
```
pip install -r scripts/requirements.txt
python3 scripts/export_vits.py
```

To build and run the program you can run: 

```
make
```
