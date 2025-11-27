# LC_MVP
Chatbot for Linde Consulting MVP

## Overview

A conversational chatbot that supports loading pretrained language models from local files or Hugging Face Hub.

## Features

- Load pretrained models from local directories
- Load models from Hugging Face Hub
- Configurable generation parameters
- Conversation history tracking
- Interactive and single-message modes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode

Run the chatbot in interactive conversation mode:

```bash
# With default model (DialoGPT-medium)
python run_chatbot.py

# With a custom configuration file
python run_chatbot.py --config config/model_config.json

# With a local pretrained model
python run_chatbot.py --model /path/to/local/model

# With a specific Hugging Face model
python run_chatbot.py --model microsoft/DialoGPT-small
```

### Single Message Mode

Send a single message and get a response:

```bash
python run_chatbot.py --single "Hello, how are you?"
```

### Using as a Module

```python
from src.chatbot import Chatbot

# Initialize with default model
chatbot = Chatbot()

# Or with a local model
chatbot = Chatbot(model_path="/path/to/your/model")

# Chat
response = chatbot.chat("Hello!")
print(response)

# Reset conversation
chatbot.reset_conversation()
```

## Loading Pretrained Models

### From Local Files

Place your model files in the `models/` directory. The directory should contain:

- `config.json` (required)
- `pytorch_model.bin` or `model.safetensors`
- `tokenizer.json` or `tokenizer_config.json`
- Additional tokenizer files as needed

Then run:
```bash
python run_chatbot.py --model models/your-model-name
```

### From Hugging Face Hub

Simply specify the model identifier:
```bash
python run_chatbot.py --model microsoft/DialoGPT-medium
```

## Configuration

Edit `config/model_config.json` to customize:

- `model_name_or_path`: Default model to load
- `local_model_path`: Path to local model (overrides model_name_or_path)
- `max_length`: Maximum response length
- `temperature`: Sampling temperature (0.0-1.0)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `device`: Device to use ("auto", "cpu", or "cuda")

## Project Structure

```
LC_MVP/
├── config/
│   └── model_config.json    # Model configuration
├── models/                   # Place local models here
├── src/
│   ├── __init__.py
│   ├── chatbot.py           # Main chatbot class
│   └── model_loader.py      # Model loading utilities
├── requirements.txt
├── run_chatbot.py           # Entry point
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full list
