# Translator - Turkish to English Seq2Seq Model

This repository contains a simple sequence-to-sequence (Seq2Seq) neural network model built with TensorFlow/Keras. It translates Turkish sentences into English using an LSTM-based encoder-decoder architecture.

## Features

- Word-level tokenization
- Encoder-Decoder structure with LSTM layers
- Greedy decoding during inference
- Custom, manually defined bilingual dataset (20 sentence pairs)
- Easily extendable to larger datasets

## How It Works

1. The Turkish sentences are tokenized and padded.
2. Target English sentences are prepared with `startseq` and `endseq` tokens.
3. The model is trained using categorical crossentropy on one-hot encoded target outputs.
4. During inference:
   - Encoder returns the initial states.
   - Decoder generates one word at a time using greedy search until `endseq` is generated.

## Model Architecture

- Encoder: Embedding + LSTM (returns final hidden and cell states)
- Decoder: Embedding + LSTM (uses encoder states) + Dense layer with softmax
- Training: Teacher forcing
- Inference: Stateful decoding with updated LSTM states

## Training

The model is trained for 600 epochs on 20 sentence pairs. You can increase the number of sentence pairs to improve performance.

```python
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_onehot,
    batch_size=8,
    epochs=600,
    verbose=1
)
