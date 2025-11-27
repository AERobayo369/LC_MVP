"""
Chatbot Module
Main chatbot class that uses pretrained models for conversation.
"""

from typing import Optional, List, Dict, Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .model_loader import ModelLoader


class Chatbot:
    """
    A conversational chatbot that uses pretrained language models.
    
    Features:
    - Load pretrained models from local files or Hugging Face Hub
    - Maintain conversation history
    - Configurable generation parameters
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        """
        Initialize the Chatbot.
        
        Args:
            config_path: Path to model configuration JSON file.
            model_path: Path to pretrained model directory or Hugging Face model ID.
            tokenizer_path: Path to tokenizer. If None, uses model_path.
        """
        self.model_loader = ModelLoader(config_path)
        self.config = self.model_loader.get_config()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.model_loader.load_model(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
        )
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, str]] = []
        self.chat_history_ids: Optional[torch.Tensor] = None
    
    def generate_response(
        self,
        user_input: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Generate a response to the user input.
        
        Args:
            user_input: The user's message.
            max_length: Maximum length of generated response.
            temperature: Sampling temperature (higher = more creative).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
        
        Returns:
            The chatbot's response as a string.
        """
        # Use config defaults if not specified (use 'is None' to handle zero values)
        if max_length is None:
            max_length = self.config.get("max_length", 200)
        if temperature is None:
            temperature = self.config.get("temperature", 0.7)
        if top_p is None:
            top_p = self.config.get("top_p", 0.9)
        if top_k is None:
            top_k = self.config.get("top_k", 50)
        do_sample = self.config.get("do_sample", True)
        
        # Encode user input
        new_user_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token,
            return_tensors='pt'
        ).to(self.model.device)
        
        # Append to chat history for context
        if self.chat_history_ids is not None:
            bot_input_ids = torch.cat(
                [self.chat_history_ids, new_user_input_ids],
                dim=-1
            )
        else:
            bot_input_ids = new_user_input_ids
        
        # Generate response using max_new_tokens instead of max_length
        # to avoid issues with growing conversation history
        with torch.no_grad():
            self.chat_history_ids = self.model.generate(
                bot_input_ids,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=1,
            )
        
        # Decode the response
        response = self.tokenizer.decode(
            self.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
            skip_special_tokens=True
        )
        
        # Store in conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        self.chat_history_ids = None
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Return the conversation history."""
        return self.conversation_history.copy()
    
    def chat(self, user_input: str) -> str:
        """
        Alias for generate_response with default parameters.
        
        Args:
            user_input: The user's message.
        
        Returns:
            The chatbot's response.
        """
        return self.generate_response(user_input)


def run_chatbot(
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> None:
    """
    Run the chatbot in interactive mode.
    
    Args:
        config_path: Path to model configuration JSON file.
        model_path: Path to pretrained model or Hugging Face model ID.
    """
    print("=" * 60)
    print("Linde Consulting MVP Chatbot")
    print("=" * 60)
    print("\nInitializing chatbot...")
    
    chatbot = Chatbot(
        config_path=config_path,
        model_path=model_path,
    )
    
    print("\nChatbot ready! Type 'quit' or 'exit' to end the conversation.")
    print("Type 'reset' to start a new conversation.")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == 'reset':
                chatbot.reset_conversation()
                print("\nConversation reset. Starting fresh!")
                continue
            
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue
