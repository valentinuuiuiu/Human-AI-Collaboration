#!/usr/bin/env python3
"""
Test script for LoRA fine-tuning setup
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

def test_model_loading():
    model_name = "ibm-granite/granite-4.0-350m"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✅ Tokenizer loaded")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Force CPU
        trust_remote_code=True
    )
    print("✅ Model loaded")

    print(f"Model parameters: {model.num_parameters()}")
    return model, tokenizer

if __name__ == "__main__":
    test_model_loading()