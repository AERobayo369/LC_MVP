"""
Model Loader Module
Handles loading of pretrained models from local files or Hugging Face Hub.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


class ModelLoader:
    """
    Loads pretrained models from local files or Hugging Face Hub.
    
    Supports loading models from:
    - Local directory containing model files (config.json, pytorch_model.bin, etc.)
    - Hugging Face Hub model identifier
    - Custom model path specified in configuration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ModelLoader.
        
        Args:
            config_path: Path to model configuration JSON file.
                        If None, uses default configuration.
        """
        self.config = self._load_config(config_path)
        self.device = self._get_device()
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from JSON file or return defaults."""
        default_config = {
            "model_name_or_path": "microsoft/DialoGPT-medium",
            "tokenizer_name_or_path": None,
            "local_model_path": None,
            "max_length": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "do_sample": True,
            "num_return_sequences": 1,
            "pad_token_id": None,
            "device": "auto",
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
        
        return default_config
    
    def _get_device(self) -> str:
        """Determine the appropriate device for model inference."""
        device_config = self.config.get("device", "auto")
        
        if device_config == "auto":
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        
        return device_config
    
    def load_model(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load a pretrained model and tokenizer.
        
        Args:
            model_path: Path to model directory or Hugging Face model ID.
                       If None, uses config setting or default.
            tokenizer_path: Path to tokenizer. If None, uses model_path.
        
        Returns:
            Tuple of (model, tokenizer)
        
        Raises:
            FileNotFoundError: If local model path is specified but doesn't exist.
            ValueError: If model cannot be loaded.
        """
        # Determine model path priority:
        # 1. Explicit parameter
        # 2. Local model path from config
        # 3. Model name/path from config
        final_model_path = (
            model_path
            or self.config.get("local_model_path")
            or self.config.get("model_name_or_path")
        )
        
        if not final_model_path:
            raise ValueError("No model path specified")
        
        # Check if it's a local path
        if os.path.isdir(final_model_path):
            return self._load_from_local(final_model_path, tokenizer_path)
        else:
            return self._load_from_hub(final_model_path, tokenizer_path)
    
    def _load_from_local(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from local directory."""
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        
        # Check for required files
        config_file = model_dir / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"config.json not found in model directory: {model_path}"
            )
        
        print(f"Loading model from local path: {model_path}")
        
        # Load tokenizer
        tokenizer_dir = tokenizer_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            local_files_only=True,
            trust_remote_code=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully on {self.device}")
        return self.model, self.tokenizer
    
    def _load_from_hub(
        self,
        model_name: str,
        tokenizer_path: Optional[str] = None,
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from Hugging Face Hub."""
        print(f"Loading model from Hugging Face Hub: {model_name}")
        
        # Load tokenizer
        tokenizer_name = tokenizer_path or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model loaded successfully on {self.device}")
        return self.model, self.tokenizer
    
    def get_model(self) -> Optional[PreTrainedModel]:
        """Return the loaded model, if any."""
        return self.model
    
    def get_tokenizer(self) -> Optional[PreTrainedTokenizer]:
        """Return the loaded tokenizer, if any."""
        return self.tokenizer
    
    def get_config(self) -> Dict[str, Any]:
        """Return the current configuration."""
        return self.config.copy()
