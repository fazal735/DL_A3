# DL_A3
Deep Learning Assignment 3 
This repository contains a neural sequence-to-sequence model for transliterating Latin script (Roman letters) to Devanagari script. The model can be used to convert English spellings of Urdu words to their proper Devanagari representation.Table of Contents

Overview

Model Architecture
Installation
Usage

Training a New Model
Using a Pre-trained Model
Integrating in Other Projects


Data
Components
Customization
Performance
Future Improvements

Overview
Transliteration is the process of converting text from one script to another while preserving the phonetic characteristics. This project provides a neural network-based solution for transliterating Latin characters (English alphabet) to Devanagari script, which is used for writing Hindi, Sanskrit, Marathi, and other languages in India.
This implementation uses a sequence-to-sequence (Seq2Seq) model with LSTMs to learn the mapping between character sequences. The model learns to generate Devanagari characters based on the input Latin characters, capturing the nuances of transliteration rules.
Model Architecture
The transliteration model uses a sequence-to-sequence architecture with the following components:
Encoder

Character-level embedding layer for Latin characters
Bidirectional LSTM layers to process the input sequence
Configurable number of layers and dropout for regularization

Decoder

Character-level embedding layer for Devanagari characters
LSTM layers with the same hidden dimension as the encoder
Linear output layer to predict the next Devanagari character

Features

Teacher forcing during training (using ground truth as input with a certain probability)
Support for variable-length input sequences through padding and packing
Early stopping to prevent overfitting
Checkpoint saving to preserve the best model


Components
The code consists of several key components:
1. CharacterEmbedding
Converts character indices to dense vectors using PyTorch's embedding layer.
2. Encoder
Processes the input Latin sequence using an RNN (LSTM/GRU).
3. Decoder
Generates Devanagari characters based on the encoder's output.
4. Seq2SeqModel
Combines the encoder and decoder into a complete sequence-to-sequence model.
5. TransliterationDataset
Handles data preparation and conversion between characters and indices.
6. Training Functions

train_model: Trains the model for one epoch
evaluate_model: Evaluates the model on validation data
transliterate: Converts a Latin string to Devanagari using the trained model

Customization
You can customize the model through several parameters:
Model Architecture

embedding_dim: Size of character embeddings (default: 128)
hidden_dim: Size of LSTM hidden states (default: 256)
encoder_layers/decoder_layers: Number of LSTM layers (default: 2)
dropout: Dropout rate for regularization (default: 0.2)
encoder_cell_type/decoder_cell_type: Type of RNN cell ('lstm', 'gru', or 'rnn')

Training Parameters

n_epochs: Maximum number of training epochs (default: 100)
clip: Gradient clipping value (default: 1.0)
teacher_forcing_ratio: Probability of using teacher forcing (default: 0.5)
patience: Number of epochs to wait before early stopping (default: 5)

Example customization:


