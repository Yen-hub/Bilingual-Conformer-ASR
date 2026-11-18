# Coding 4: Speech Recognition

Conformer-based attention encoder-decoder for English and Italian automatic speech recognition.

## Contents

- `train.py` and `decode.py` provide the training and decoding entry points.
- `models/` contains the frontend, encoder, decoder, and neural network layers.
- `loader.py` handles batching, tokenization, and multilingual language tags.
- `bpe.model` and `dict.txt` provide the tokenizer model and vocabulary.
- `conf/base.yaml` contains the example model and training configuration.

The dataset is not included. Follow the course handout for environment and dataset setup before training.
