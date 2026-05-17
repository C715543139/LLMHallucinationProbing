"""Test: re-load model with eager attention for attention extraction only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

print("=== Test: Re-load model with eager from scratch ===")
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models_cache/Qwen2-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load with eager attention from the start
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="eager",
)
model.eval()
print(f"Model loaded, attn_impl={model.config.attn_implementation}")

# Test with padding
statements = ["Hello world", "The sky is blue.", "Paris is in France."]
inputs = tokenizer(statements, return_tensors="pt", padding=True, truncation=True, max_length=128)
device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    o = model(**inputs, output_hidden_states=True, output_attentions=True)

hs = o.hidden_states[18]
attn = o.attentions[17]
print(f"Hidden states: shape={hs.shape}, NaN={torch.isnan(hs).sum().item()}, mean={hs.mean().item():.4f}")
print(f"Attention[17]: shape={attn.shape}, NaN={torch.isnan(attn).sum().item()}, mean={attn.mean().item():.4f}")

# Second pass
with torch.no_grad():
    o2 = model(**inputs, output_hidden_states=True, output_attentions=True)
hs2 = o2.hidden_states[18]
print(f"Second pass Hidden: NaN={torch.isnan(hs2).sum().item()}, mean={hs2.mean().item():.4f}")

# Check attention mask handling
print(f"\nAttention mask: {inputs['attention_mask']}")
print(f"Last token positions: {inputs['attention_mask'].sum(dim=1) - 1}")
last_hidden = hs[torch.arange(3), inputs['attention_mask'].sum(dim=1) - 1]
print(f"Last token hidden stats: NaN={torch.isnan(last_hidden).sum().item()}, mean={last_hidden.mean().item():.4f}")
print("DONE")
