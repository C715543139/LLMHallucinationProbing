"""
Eager attention 诊断脚本：验证当前环境下 Qwen2 的 output_attentions 路径是否稳定，
并区分单样本、带 padding 的 batch、逐样本提取三种典型调用方式。

用法:
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate llm_hallucination
    source ./.venv/bin/activate
    python -s scripts/diagnose_eager.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models_cache/Qwen2-1.5B"

def load_eager():
    """加载模型然后切换到 eager attention。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    # 关键：加载后切换
    model.set_attn_implementation("eager")
    attn_impl = getattr(
        model.config,
        "attn_implementation",
        getattr(model.config, "_attn_implementation", "unknown"),
    )
    print(f"  attn_implementation = {attn_impl}")
    return model, tokenizer

def check_nan(tensor, name):
    n = torch.isnan(tensor).sum().item()
    total = tensor.numel()
    status = "NaN" if n > 0 else "OK"
    print(f"  {name}: shape={tensor.shape}, NaN={n}/{total} {status}")
    return n == 0

print("=" * 60)
print("  Eager Attention 诊断")
print("=" * 60)

model, tokenizer = load_eager()
device = next(model.parameters()).device

# ---- Test 1: 单样本，无 padding ----
print("\n[Test 1] 单样本，无 padding")
s1 = "Paris is the capital of France."
inp1 = tokenizer(s1, return_tensors="pt").to(device)
with torch.no_grad():
    o1 = model(**inp1, output_hidden_states=True, output_attentions=True)
ok1_h = check_nan(o1.hidden_states[18], "hidden[18]")
ok1_a = check_nan(o1.attentions[17], "attn[17]")
print(f"  结论: {'单样本无padding正常' if (ok1_h and ok1_a) else '单样本也异常'}")

# ---- Test 2: 多样本，有 padding ----
print("\n[Test 2] 多样本，有 padding")
statements = ["Hello.", "The sky is blue.", "A longer sentence for testing purposes."]
inp2 = tokenizer(statements, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    o2 = model(**inp2, output_hidden_states=True, output_attentions=True)
ok2_h = check_nan(o2.hidden_states[18], "hidden[18]")
ok2_a = check_nan(o2.attentions[17], "attn[17]")

# ---- Test 3: 单样本逐个处理（模拟 batch_size=1 的 attention 提取） ----
print("\n[Test 3] 多样本逐个处理（无 padding）")
all_ok = True
for i, s in enumerate(statements):
    inp = tokenizer(s, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model(**inp, output_hidden_states=True, output_attentions=True)
    ok = check_nan(o.hidden_states[18], f"  sample[{i}] hidden[18]")
    ok &= check_nan(o.attentions[17], f"  sample[{i}] attn[17]")
    all_ok &= ok
print(f"  结论: {'逐个处理正常' if all_ok else '逐个处理也异常'}")

# ---- Test 4: 单样本 + attention score 特征提取函数 ----
print("\n[Test 4] 使用 extract_attention_score_features_single")
from src.features.attention_scores import extract_attention_score_features_single
feats, names, meta = extract_attention_score_features_single(
    model, tokenizer, "Paris is the capital of France.", [0, 1, 2])
n_nan = np.isnan(feats).sum()
print(f"  features: shape={feats.shape}, NaN={n_nan}/{len(feats)} {'OK' if n_nan==0 else 'NaN'}")

# ---- Test 5: 验证 output_attentions 在 eager 下是否返回非 None ----
print("\n[Test 5] output_attentions 是否正常返回")
inp5 = tokenizer("Test.", return_tensors="pt").to(device)
with torch.no_grad():
    o5 = model(**inp5, output_attentions=True)
if o5.attentions is None:
    print("  attentions is None - eager attention 未生效!")
elif len(o5.attentions) == 0:
    print("  attentions is empty!")
else:
    print(f"  attentions: {len(o5.attentions)} layers, shape={o5.attentions[0].shape}")

print("\n" + "=" * 60)
print("  诊断完成")
print("=" * 60)
