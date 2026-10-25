# TinyGPT

A GPT-style decoder-only Transformer implemented from scratch in PyTorch and trained on the TinyStories dataset for next-token prediction and autoregressive text generation.

This repository extends my previous project, Transformer-from-Scratch, by moving from a toy implementation to a complete language modeling pipeline. While the first project focused on implementing and understanding individual Transformer components, this project brings them together into a GPT-style language model that can learn from real text and generate new sequences autoregressively.

I conducted five experiments to investigate the impact of different model architectures, dataset sizes, and training configurations. The experiments progressively scale from a small model trained on 10,000 characters to larger models trained on up to 2 million characters from TinyStories.

The implementation remains intentionally compact, modular, and educational, with the core Transformer components implemented from scratch using PyTorch.

# Project Overview

<pre>
Tiny Stories
        │
        ▼
Vocabulary Construction
        │
        ▼
Character Tokenization
        │
        ▼
Training / Validation Split
        │
        ▼
Mini-batch Generation
        │
        ▼
Token Embeddings
        │
        ▼
Positional Embeddings
        │
        ▼
Transformer Blocks
   ├── Multi-Head Attention
   ├── Feed Forward Network
   ├── Layer Normalization
   └── Residual Connections
        │
        ▼
Language Modeling Head
        │
        ▼
Cross-Entropy Loss
        │
        ▼
AdamW Optimization
        │
        ▼
Autoregressive Text Generation
</pre>

#  Repository Structure

<pre>
Tiny_GPT/
│
├── src/
│   ├── data.py              # Data loading, vocabulary construction,
│   │                       # character tokenization, dataset splitting,
│   │                       # and mini-batch generation.
│   │
│   ├── attention.py         # Single-head and multi-head causal
│   │                       # self-attention implementations.
│   │
│   ├── transformer.py       # Transformer block combining attention,
│   │                       # MLP, layer normalization, and residual connections.
│   │
│   ├── model.py             # TinyGPT architecture, next-token prediction,
│   │                       # and autoregressive text generation.
│   │
│   └── train.py             # Training loop, validation, and loss tracking.
│
├── tests/
│   ├── test_data.py         # Tests vocabulary construction and mini-batch generation.
│   │
│   ├── test_attention.py    # Tests attention output shapes, causal masking,
│   │                       # and gradient computation.
│   │
│   ├── test_transformer.py  # Tests Transformer block output shapes,
│   │                       # causality, and gradients.
│   │
│   ├── test_tiny_gpt.py     # Tests GPT output shapes, causality,
│   │                       # gradients, and generation.
│   │
│   └── test_train.py        # Tests training, evaluation, parameter updates,
│                           # and the model's ability to learn.
│
├── configs/
│   ├── experiment_01.json  # Small model and small dataset.
│   ├── experiment_02.json  # Increased dataset size.
│   ├── experiment_03.json  # Larger dataset and model.
│   ├── experiment_04.json  # Further model and dataset scaling.
│   └── experiment_05.json  # Larger configuration with extended training.
│
├── experiments/            # Saved experimental results and outputs.
│
└── run_experiments.py       # Runs experiments from JSON configurations.  
</pre>


# Features

This implementation includes
- Character-level tokenization
- Token and positional embeddings
- Causal self-attention
- Multi-head attention
- Feed-forward networks
- Residual connections
- Layer normalization
- Dropout regularization
- Training and validation loop
- Temperature-based text generation
- Modular implementation

# Improvements over my previous repository

Compared to my first repository (Transformer-from-Scratch), this implementation introduces several components used in modern GPT-style language models:

| Previous repository              | This repository                 |
| -------------------------------- | ------------------------------- |
| Toy dataset                      | Tiny Stories        |
| Basic Transformer implementation | Complete GPT training pipeline  |
| Simple generation                | Temperature-controlled sampling |
| Single training script           | Modular project structure       |
| No regularization                | Dropout                         |
| Basic evaluation                 | Training and validation losses  |

# Running the Project

Experiments are configured through JSON files, allowing model architecture, dataset size, and training hyperparameters to be modified without changing the source code.

To run an experiment, execute the following command from the project root:

python3 run_experiments.py --config configs/experiment_01.json

Replace experiment_01.json with the configuration file corresponding to the experiment you want to run.

The five configurations explore progressively larger datasets, model capacities, and training durations. Experimental outputs are saved in the experiments/ directory.

Text Generation

An autoregressive generation function is implemented in src/model.py. Given an initial text prompt, the model generates additional tokens sequentially, using its previous predictions as context.

Generation supports temperature-based sampling to control the randomness of the generated text.

Running Tests

The repository also includes unit tests for data preprocessing, attention, Transformer blocks, the complete GPT model, training, evaluation, and text generation.

To run all tests:

python3 -m pytest tests/ -v



