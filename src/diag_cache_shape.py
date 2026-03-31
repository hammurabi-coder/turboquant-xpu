"""
Check tqc_cache shape consistency between prefill and decode tokens.
Run 2 decode steps, print shapes after each step.
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

head_dim = model.model.layers[0].self_attn.head_dim
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Intercept — print after each step
_orig = model.model.layers[0].self_attn.forward
step_count = [0]

def intercept(*args, **kwargs):
    out, _ = _orig(*args, **kwargs)
    step_count[0] += 1
    s = step_count[0]
    # After this forward call completes, tqc_cache has the new token appended
    cache = tqc_cache[0][0]  # layer 0, head 0
    n = len(cache)
    prefill_shape = cache[0][0].pq.indices.shape  # first token (prefill)
    last_shape    = cache[-1][0].pq.indices.shape  # last token (new)
    print(f"\n=== After step {s} ===", flush=True)
    print(f"  len(tqc_cache[0][0]) = {n}", flush=True)
    print(f"  prefill token indices shape: {prefill_shape}", flush=True)
    print(f"  step-{s} token indices shape: {last_shape}", flush=True)
    print(f"  MATCH? {prefill_shape == last_shape}", flush=True)
    # Also check norm shapes
    prefill_norm = cache[0][0].pq.norm.shape
    last_norm    = cache[-1][0].pq.norm.shape
    print(f"  prefill norm shape: {prefill_norm}", flush=True)
    print(f"  step-{s} norm shape: {last_norm}", flush=True)
    print(f"  NORM MATCH? {prefill_norm == last_norm}", flush=True)
    return out, None

model.model.layers[0].self_attn.forward = intercept

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=2,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nOutput: {gen!r}")
unhook(model)
