# tinyLM

**tinyLM** is a barebones transformer model implemented in PyTorch, designed for **experimental and research purposes**.  

## Overview

- This project provides a minimal transformer architecture for testing ideas and experimenting with language models.
- It does **not include optimizations** such as FlashAttention or other memory-efficient attention implementations.
- It is **not production-ready** and intended purely for **experimental use**.

## Features

- Decoder-only transformer architecture
- Configurable number of layers, hidden size, and attention heads
- Simple training loop for testing model behavior

## Usage

1. Prepare your dataset in the expected format. (default is set to SlimPajama-6B)
2. Configure model hyperparameters in `main.py` or the config file.
3. Train the model with:

```bash
python3 main.py
