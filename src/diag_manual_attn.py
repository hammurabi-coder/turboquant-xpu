"""
Manual attention recomputation at step 2 to pinpoint where output diverges.
1. Capture Q from the hook's decode step
2. Decode all cached K/V from tqc_cache
3. Compute attention scores manually
4. Compare to what the hook returned
"""
import sys, torch, math

_VRAM_TOTAL = 12 * 1024**3
def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "qwen3_modeling",
    "/home/hermes/.battlemage-venv/lib/python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py"
)
qwen3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qwen3_module)
apply_rotary_pos_emb = qwen3_module.apply_rotary_pos_emb

from transformers import AutoModelForCausalLM, AutoTokenizer
from cache import TurboQuantConfig, turboquant_decode_single
from hf_hook import install_turboquant_hook, unhook

device = torch.device("xpu")
PROMPT = "The northern kingdom was a place of"

print("Loading Qwen/Qwen3-0.6B...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map={"": "xpu"},
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
model.eval()
print("Model loaded.\n", flush=True)

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    pkv = model(ids, output_hidden_states=False).past_key_values
print(f"Prefill done, pkv tokens: {pkv.layers[0].keys.shape[2]}")

head_dim = model.model.layers[0].self_attn.head_dim
num_q_heads = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
kv_groups = num_q_heads // num_kv_heads
print(f"Model: {num_q_heads}Q / {num_kv_heads}KV heads, head_dim={head_dim}, groups={kv_groups}")

tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Also capture the raw pkv keys for baseline comparison
with torch.no_grad():
    baseline_pkv_keys = pkv.layers[0].keys.clone()  # [B, num_kv_heads, seq, D]
print(f"Baseline pkv keys shape: {baseline_pkv_keys.shape}")

# Intercept layer 0 to capture Q and the hook's attention output
_orig_fwd = model.model.layers[0].self_attn.forward
captured = {}

step_count = [0]
def intercept(*args, **kwargs):
    out, _ = _orig_fwd(*args, **kwargs)
    step_count[0] += 1
    s = step_count[0]
    captured['step'] = s
    captured['attn_out'] = out.detach().clone()
    captured['q_proj_out'] = None  # can't easily capture Q here
    return out, None

model.model.layers[0].self_attn.forward = intercept

# Run 2 decode steps
with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=2,
                              do_sample=False, pad_token_id=tok.pad_token_id)

s = captured['step']
attn_out_hooked = captured['attn_out']
print(f"\nRan {s} decode steps, last hook attn_out mean={attn_out_hooked.mean():.6f}  std={attn_out_hooked.std():.6f}")
print(f"Output: {tok.decode(out_ids[0], skip_special_tokens=True)!r}")

# Now manually compute attention for step 2 using the tqc_cache
print(f"\n{'='*60}")
print("MANUAL ATTENTION RECOMPUTATION for step 2")
print(f"{'='*60}")

cache = tqc_cache[0][0]  # layer 0, head 0
seq_len = len(cache)
print(f"Cache has {seq_len} tokens (layer 0, head 0)")

# Decode all cached K and V
k_list = []
v_list = []
for t in range(seq_len):
    k_t = turboquant_decode_single(cache[t][0])  # [D]
    v_t = turboquant_decode_single(cache[t][1])
    k_list.append(k_t)
    v_list.append(v_t)

k_decoded = torch.stack(k_list, dim=0).float()  # [seq, D]
v_decoded = torch.stack(v_list, dim=0).float()  # [seq, D]
print(f"Decoded K shape: {k_decoded.shape}, V shape: {v_decoded.shape}")

# Get Q for step 2 — re-derive from model
# Step 2 corresponds to position 7 (0-indexed), which is past_seen_tokens=7
# Run the model forward for one token to get Q at position 7
new_token_id = out_ids[0, 7].item()  # the token generated at step 2
print(f"Step 2 generated token id: {new_token_id} = {tok.decode([new_token_id])!r}")

# Get the new token embedding
new_token_emb = model.model.embed_tokens(torch.tensor([[new_token_id]], device=device))
# Get position ids for the new token
position_ids = torch.tensor([[7]], device=device)  # position 7

# Compute Q for layer 0, head 0 at position 7
attn = model.model.layers[0].self_attn
hidden_shape = (1, -1, head_dim)

# Q path
q_proj_out = attn.q_proj(new_token_emb)
q_states = attn.q_norm(q_proj_out.view(hidden_shape)).transpose(1, 2)  # [1, H, 1, D]

# Get RoPE embeddings for position 7
position_embeddings = model.model.rotary_emb(new_token_emb, position_ids)
cos, sin = position_embeddings
# Apply RoPE to Q
q_rope, _ = apply_rotary_pos_emb(q_states, None, cos, sin, unsqueeze_dim=1)
# q_rope for head 0: [1, 1, 1, D]
q_head0 = q_rope[0, 0, 0, :].float()  # [D]

print(f"Q for step 2, head 0: norm={q_head0.norm().item():.4f}")

# Compute attention scores: k_decoded @ q_head0
scale = head_dim ** -0.5
scores = (k_decoded @ q_head0) * scale  # [seq]
weights = torch.softmax(scores, dim=0)  # [seq]
print(f"\nAttention scores (k_decoded @ q): {scores.detach().cpu().tolist()}")
print(f"Softmax weights: {weights.detach().cpu().tolist()}")

# Which token has highest weight?
top_weight, top_idx = weights.max(dim=0)
print(f"Highest attention weight: {top_weight.item():.4f} on token {top_idx.item()}")

# Manual attention output
attn_manual = (weights.unsqueeze(1) * v_decoded).sum(dim=0)  # [D]
print(f"Manual attn_out norm: {attn_manual.norm().item():.4f}")

# Compare to the hooked model's attn_out for layer 0, head 0
# The hook returns attn_output after o_proj, but we need the pre-o_proj value
# Let me just compare the final attn_output
print(f"\nHook's final attn_out mean={attn_out_hooked.mean().item():.6f}  std={attn_out_hooked.std():.6f}")

# For a more direct comparison, look at the RMS of the difference
# between baseline pkv attention and tqc attention at step 2
# But we don't have baseline attn_out captured...

# Instead: decode the K/V from baseline pkv for comparison
print(f"\n--- Baseline K (from pkv) at step 2 for comparison ---")
baseline_k = baseline_pkv_keys[0, 0, :, :]  # [seq, D] — head 0, all positions
print(f"Baseline K norms per position:")
for pos in range(baseline_k.shape[0]):
    print(f"  pos {pos}: norm={baseline_k[pos].norm().item():.4f}")

# Compare baseline K norms to tqc K norms
print(f"\nTQC K norms per token:")
for t in range(seq_len):
    print(f"  token {t}: norm={k_list[t].norm().item():.4f}")

unhook(model)
