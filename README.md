# 🎵 Music Generation with Recurrent Neural Networks (LSTM)

**Module:** IA générative et ingénierie des prompts
**Program:** Master SDIA
**Student:** **SALMA LIDAME**

## Project Overview

This project explores **music generation using Recurrent Neural Networks (RNNs)**, specifically **LSTM architectures**, applied to symbolic music represented in **ABC notation**.

The goal is to train a character-level generative model capable of learning musical structure and generating new melodies by predicting the next character in a sequence.

The project covers the **full deep learning pipeline**:

* Data exploration and preprocessing
* Character-level sequence modeling
* LSTM architecture design
* Training optimization and feasibility analysis
* Autoregressive music generation

##  Objectives

* Model musical sequences as text using **ABC notation**
* Learn long-term musical dependencies with **LSTM**
* Implement a **character-level generative model**
* Analyze computational constraints and optimize training
* Explore creativity vs coherence using **temperature sampling**


##  Dataset

* **Irish Traditional Music Dataset**
* Format: JSON files containing full ABC partitions
* Files:

  * `train.json`: 214,122 songs
  * `validation.json`: 2,162 songs
* Train/Validation ratio: **~99:1**

Each song is represented as a **raw text sequence** describing melody, meter, key, and structure.


##  ABC Notation

ABC is a compact ASCII-based music notation:

* Notes: `A–G`
* Durations: `2`, `/2`
* Measures: `|`
* Repetitions: `:`, `::`
* Metadata:

  * `X:` index
  * `T:` title
  * `M:` meter
  * `K:` key

This format is ideal for sequence modeling with RNNs.


##  Data Preprocessing

* **Character-level modeling**
* Vocabulary size: **95 unique characters**
* Character ↔ index mapping (`char_to_idx`, `idx_to_char`)
* Vectorization of songs into integer sequences
* **Padding strategy** to handle variable-length sequences

### Sequence Statistics

* Min length: 22 characters
* Median length: 257
* Mean length: 290
* Max length: 2,968


## Model Architecture

### MusicRNN (LSTM-based)

* **Embedding dimension:** 256
* **Hidden size:** 512 → 1024 (initial experiments)
* **Number of layers:** 2–3 stacked LSTMs
* **Dropout:** 0.3
* **Vocabulary size:** 95

**Total parameters:** ~3.75M

Why LSTM?

* Handles long-term dependencies
* Mitigates vanishing gradients
* Suitable for long sequential data like music


##  Training Strategy

* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam / AdamW
* **Learning rate:** 5e-3 → 5e-4
* **Batch size:** 64–256
* **Early stopping** to prevent overfitting
* **TensorBoard** for monitoring loss and accuracy
* **Gradient clipping** for stability


## Computational Analysis

Initial configuration was **computationally infeasible**:

* ~350 days of training time
* ~15–20 GB VRAM required
* Not compatible with Kaggle P100 constraints


## Optimized Configuration

To make training feasible:

* We Reduced dataset size:

  * Train: 20,000 songs
  * Validation: 2,000 songs
* Truncated sequences to **max_length = 600**

  * Covers ~95% of songs
* Optimized hyperparameters:

  * Hidden size: 768
  * Layers: 3
  * AdamW + LR scheduler
  * Training epochs: 35

This configuration balances **performance, creativity, and feasibility**.

# Performance: 
Val Accuracy: 0.8719

## 🎶 Music Generation

The model generates music **autoregressively**:

1. Start from a seed sequence
2. Predict next character probabilities
3. Sample using **temperature**
4. Append character and repeat

### Temperature Sampling

* `T < 1` → more deterministic
* `T = 1` → balanced
* `T > 1` → more creative, less stable

Generated outputs can be played using **ABC players**.
example : 
<img width="1246" height="322" alt="one TUNE" src="https://github.com/user-attachments/assets/10eea242-f103-479c-b36c-1fcfc64cc63b" />



##  References

* PyTorch Documentation
* TensorBoard
* ABC Notation


