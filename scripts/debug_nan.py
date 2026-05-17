"""Quick debug: check hidden states extraction with eager attention."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loader import load_model_fp16
from src.features.hidden_states import extract_hidden_states
import torch
import numpy as np

print("Loading model...")
model, tokenizer = load_model_fp16(model_path="models_cache/Qwen2-1.5B")
model.set_attn_implementation("eager")
print(f"Model device: {next(model.parameters()).device}")

# Test 1: single statement via extract_last_token_hidden
from src.features.hidden_states import extract_last_token_hidden
h1 = extract_last_token_hidden(model, tokenizer, "Hello world", layer_idx=17)
print(f"Test 1 (single, L17 last): shape={h1.shape}, NaN={np.isnan(h1).sum()}, mean={np.nanmean(h1):.4f}")

# Test 2: batch via extract_hidden_states
statements = ["Hello world", "The sky is blue.", "Paris is in France."]
h2 = extract_hidden_states(model, tokenizer, statements, layers=[17], pooling="last", batch_size=2)
print(f"Test 2 (batch, L17 last): shape={h2.shape}, NaN={np.isnan(h2).sum()}, mean={np.nanmean(h2):.4f}")

# Test 3: check if output_attentions causes the NaN
device = next(model.parameters()).device
inputs = tokenizer(statements, return_tensors="pt", padding=True, truncation=True, max_length=128)
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    o1 = model(**inputs, output_hidden_states=True)
    o2 = model(**inputs, output_hidden_states=True, output_attentions=True)

hs1 = o1.hidden_states[18]  # layer 17 (0=embedding, 1-28=blocks)
hs2 = o2.hidden_states[18]
print(f"Test 3a (no attn): shape={hs1.shape}, NaN={torch.isnan(hs1).sum().item()}, mean={hs1.mean().item():.4f}")
print(f"Test 3b (with attn): shape={hs2.shape}, NaN={torch.isnan(hs2).sum().item()}, mean={hs2.mean().item():.4f}")
print("DONE")
