"""
Compare actual decoded K vectors between prefill tokens and new decode tokens.
Decode from PQ, compute norms and dot products to see if they're genuinely similar.
"""
import sys, torch

_VRAM_TOTAL = 12 * 1024**3
def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

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
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Intercept to capture tqc_cache state AFTER each decode step
_orig = model.model.layers[0].self_attn.forward
step_count = [0]

def intercept(*args, **kwargs):
    out, _ = _orig(*args, **kwargs)
    step_count[0] += 1
    s = step_count[0]
    cache = tqc_cache[0][0]  # layer 0, head 0

    print(f"\n{'='*60}")
    print(f"STEP {s} — layer 0, head 0 — decoded K comparison")
    print(f"{'='*60}")
    print(f"Cache has {len(cache)} tokens")

    # Decode and compare: token 1 vs token 8 (suspected duplicate)
    if s >= 3 and len(cache) >= 9:
        # Compare all pairs
        print("\n--- Per-token decoded K norms ---")
        for t in range(len(cache)):
            k_dec = turboquant_decode_single(cache[t][0])
            k_norm = k_dec.norm().item()
            print(f"  token {t}: decoded K norm={k_norm:.4f}")

        # Compare token 1 and token 8 (step 2 and step 3 new tokens)
        print("\n--- Comparing token 1 (prefill) vs token 8 (step-3 new) ---")
        k1 = turboquant_decode_single(cache[1][0])
        k8 = turboquant_decode_single(cache[8][0])
        cos_sim = torch.nn.functional.cosine_similarity(k1.flatten(), k8.flatten(), dim=0).item()
        diff = (k1 - k8).abs().mean().item()
        print(f"  Token 1 norm: {k1.norm().item():.4f}")
        print(f"  Token 8 norm: {k8.norm().item():.4f}")
        print(f"  Cosine similarity: {cos_sim:.6f}")
        print(f"  Mean abs diff: {diff:.6f}")

    elif s >= 1:
        print("\n--- Per-token decoded K norms ---")
        for t in range(len(cache)):
            k_dec = turboquant_decode_single(cache[t][0])
            print(f"  token {t}: decoded K norm={k_dec.norm().item():.4f}")

    return out, None

model.model.layers[0].self_attn.forward = intercept

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=3,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nOutput: {gen!r}")
unhook(model)
