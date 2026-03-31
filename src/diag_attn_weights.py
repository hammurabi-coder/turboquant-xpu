"""
Print per-token attention weights for layer 0, head 0 at each decode step.
This shows which cached token the model attends to most.
"""
import sys, torch

_VRAM_TOTAL = 12 * 1024**3
def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

from transformers import AutoModelForCausalLM, AutoTokenizer
from cache import TurboQuantConfig
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
print(f"Prompt tokens: {tok.decode(ids[0])}")
print()

head_dim = model.model.layers[0].self_attn.head_dim
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Monkey-patch turboquant_decode_single to capture decode details
import cache as cache_module
_orig_decode = cache_module.turboquant_decode_single
_decode_info = {}

def patched_decode(ck, layer_idx=0, head_idx=0, step=0):
    result = _orig_decode(ck)
    # Store for inspection
    _decode_info[(layer_idx, head_idx, step)] = result.detach().clone()
    return result

cache_module.turboquant_decode_single = patched_decode

# Also patch to capture scores per step
_orig_layer0_fwd = model.model.layers[0].self_attn.forward
step_count = [0]
q_at_step = {}

def intercept(*args, **kwargs):
    out, _ = _orig_layer0_fwd(*args, **kwargs)
    step_count[0] += 1
    s = step_count[0]

    # After hook runs (the hooked forward is what was just called),
    # we can peek at tqc_cache to compute manual attention weights
    # for layer 0, head 0 and compare
    cache = tqc_cache[0][0]  # layer 0, head 0
    seq_len = len(cache)
    print(f"\n=== Step {s} — layer 0 head 0 attention ===", flush=True)
    print(f"  Cache has {seq_len} tokens", flush=True)

    # Decode all cached K and compute softmax scores for Q
    if seq_len > 0 and s >= 1:
        # Decode all cached K and show norms for ALL tokens
        print(f"  Cached K norms (all {seq_len} tokens):", flush=True)
        for t in range(seq_len):
            norm_val = cache[t][0].pq.norm.item()
            idx_sample = cache[t][0].pq.indices[0, :3].tolist()
            print(f"    token {t}: norm={norm_val:.4f} idx[:3]={idx_sample}", flush=True)

    print(f"  attn_out mean={out.mean().item():+.6f}  std={out.std():.6f}", flush=True)
    return out, None

model.model.layers[0].self_attn.forward = intercept

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=3,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nOutput: {gen!r}")
unhook(model)
