"""
Diagnostic: Compare attention output scale between baseline and hooked model.
Run 10-token prompt, generate 5 tokens, track layer-0 attn output per step.

The interceptor intercepts at the self_attn.forward level.
Qwen2Attention.forward returns: (attn_output: [B, seq, hidden_dim], attn_weights: None)
We intercept and measure attn_output stats, then return (attn_output, None).
"""
import sys
import torch

# --- VRAM patch ---
_VRAM_TOTAL = 12 * 1024**3
def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from cache import TurboQuantConfig, TurboQuantCache
from hf_hook import install_turboquant_hook, unhook

device = torch.device("xpu")
PROMPT = "The northern kingdom was a place of"
MAX_NEW_TOKENS = 5

# ── Load model ────────────────────────────────────────────────────────────────
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

################################################################################
# BASELINE (no hook)
################################################################################
print("=" * 60)
print("BASELINE (no hook)")
print("=" * 60)

attn_stats = []
# Save the original forward
_orig_attn_forward = model.model.layers[0].self_attn.forward

def baseline_intercept(*args, **kwargs):
    """Intercept layer-0 self_attn. Measure output, pass through unchanged."""
    attn_output, attn_weights = _orig_attn_forward(*args, **kwargs)
    # attn_output: [B, seq, hidden_dim]
    stats = {
        "mean": attn_output.mean().item(),
        "std": attn_output.std().item(),
        "max_abs": attn_output.abs().max().item(),
        "shape": tuple(attn_output.shape),
    }
    attn_stats.append(stats)
    step = len(attn_stats)
    print(f"  [Baseline step {step}] mean={stats['mean']:+.6f}  std={stats['std']:.6f}  |max|={stats['max_abs']:.6f}  shape={stats['shape']}",
          flush=True)
    return attn_output, attn_weights

model.model.layers[0].self_attn.forward = baseline_intercept

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
print(f"Input: {tok.decode(ids[0])!r}\n")

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen_text = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nBaseline output: {gen_text!r}")
baseline_output = gen_text

# Top-5 logits at step 1
model.model.layers[0].self_attn.forward = _orig_attn_forward
ids_slim = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    result = model.generate(ids_slim, max_new_tokens=1, do_sample=False,
                              pad_token_id=tok.pad_token_id,
                              output_scores=True, return_dict_in_generate=True)
    logits_step1 = result.scores[0][0]
    top5_vals, top5_idxs = logits_step1.topk(5)
    baseline_top5 = [(v.item(), idx.item(), tok.decode([idx.item()])) for v, idx in zip(top5_vals, top5_idxs)]
    print(f"\nBaseline step-1 top-5 logits:")
    for i, (v, tid, tstr) in enumerate(baseline_top5):
        print(f"  {i+1}. val={v:+.4f}  id={tid:>6}  text={tstr!r}")

baseline_attn_stats = list(attn_stats)
model.model.layers[0].self_attn.forward = baseline_intercept  # re-arm for hooked section

################################################################################
# HOOKED (TurboQuant)
################################################################################
print("\n" + "=" * 60)
print("HOOKED (TurboQuant)")
print("=" * 60)

head_dim = model.model.layers[0].self_attn.head_dim
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

# Prefill
ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    pkv = model(ids, output_hidden_states=False).past_key_values
print(f"Prefill done, pkv tokens: {pkv.layers[0].keys.shape[2]}\n")

# Install hook (replaces all self_attn.forwards)
tqc_cache = install_turboquant_hook(model, tqc_config, pkv)

# Save reference to the hook's forward (before we replace it)
_hooked_attn_forward = model.model.layers[0].self_attn.forward

# Peek at what the hook returns per decode step
hooked_stats = []

def hooked_intercept(*args, **kwargs):
    """Call the installed hook, then measure its output."""
    attn_output, attn_weights = _hooked_attn_forward(*args, **kwargs)
    # attn_output: [B, seq, hidden_dim] — same as baseline
    stats = {
        "mean": attn_output.mean().item(),
        "std": attn_output.std().item(),
        "max_abs": attn_output.abs().max().item(),
        "shape": tuple(attn_output.shape),
    }
    hooked_stats.append(stats)
    step = len(hooked_stats)
    print(f"  [Hooked step {step}] mean={stats['mean']:+.6f}  std={stats['std']:.6f}  |max|={stats['max_abs']:.6f}  shape={stats['shape']}",
          flush=True)
    return attn_output, attn_weights

model.model.layers[0].self_attn.forward = hooked_intercept

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen_text_hooked = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nHooked output: {gen_text_hooked!r}")

# Top-5 logits at step 1
unhook(model)
model.model.layers[0].self_attn.forward = _orig_attn_forward
ids_slim = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    pkv_slim = model(ids_slim, output_hidden_states=False).past_key_values
    install_turboquant_hook(model, tqc_config, pkv_slim)
    result = model.generate(ids_slim, max_new_tokens=1, do_sample=False,
                              pad_token_id=tok.pad_token_id,
                              output_scores=True, return_dict_in_generate=True)
    logits_step1 = result.scores[0][0]
    top5_vals, top5_idxs = logits_step1.topk(5)
    hooked_top5 = [(v.item(), idx.item(), tok.decode([idx.item()])) for v, idx in zip(top5_vals, top5_idxs)]
    print(f"\nHooked step-1 top-5 logits:")
    for i, (v, tid, tstr) in enumerate(hooked_top5):
        print(f"  {i+1}. val={v:+.4f}  id={tid:>6}  text={tstr!r}")

unhook(model)

################################################################################
# SUMMARY TABLE
################################################################################
print("\n" + "=" * 60)
print("SUMMARY: Layer-0 attention output per decode step")
print("=" * 60)
print(f"{'Step':>4} | {'Baseline mean':>14} {'Baseline std':>12} {'Baseline |max|':>13} | {'Hooked mean':>14} {'Hooked std':>12} {'Hooked |max|':>13}")
print("-" * 95)
for i, (bl, hk) in enumerate(zip(baseline_attn_stats, hooked_stats)):
    print(f"{i+1:>4} | {bl['mean']:>+14.6f} {bl['std']:>12.6f} {bl['max_abs']:>13.6f} | "
          f"{hk['mean']:>+14.6f} {hk['std']:>12.6f} {hk['max_abs']:>13.6f}")

print("\n" + "=" * 60)
print("DIAGNOSIS HINTS")
print("=" * 60)
print("If hooked values are ALL NEAR ZERO  → PQ decode produced garbage")
print("If hooked SCALE is wrong (e.g. 10x) → precision loss or wrong codebook")
print("If hooked LOGITS differ completely  → RoPE mismatch or K encode bug")
print("If hooked LOGITS are shifted        → codebook index bit-error")
print()
print(f"Baseline output: {baseline_output!r}")
print(f"Hooked   output: {gen_text_hooked!r}")
