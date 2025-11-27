#!/usr/bin/env python3
"""
Main entry point for the LC_MVP Chatbot.
Run the chatbot with pretrained models from local files or Hugging Face Hub.
"""

import argparse
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chatbot import Chatbot, run_chatbot


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Linde Consulting MVP Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model (DialoGPT-medium from Hugging Face)
  python run_chatbot.py

  # Run with a custom config file
  python run_chatbot.py --config config/model_config.json

  # Run with a local pretrained model
  python run_chatbot.py --model /path/to/local/model

  # Run with a specific Hugging Face model
  python run_chatbot.py --model microsoft/DialoGPT-small
        """,
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to model configuration JSON file",
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to pretrained model directory or Hugging Face model ID",
    )
    
    parser.add_argument(
        "--tokenizer", "-t",
        type=str,
        default=None,
        help="Path to tokenizer (if different from model)",
    )
    
    parser.add_argument(
        "--single", "-s",
        type=str,
        default=None,
        help="Single message mode: send one message and exit",
    )
    
    args = parser.parse_args()
    
    # Single message mode
    if args.single:
        chatbot = Chatbot(
            config_path=args.config,
            model_path=args.model,
            tokenizer_path=args.tokenizer,
        )
        response = chatbot.chat(args.single)
        print(response)
        return
    
    # Interactive mode
    run_chatbot(
        config_path=args.config,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
