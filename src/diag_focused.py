"""
Focused diagnostic: monkey-patch turboquant_encode_internal to print
the raw k_vec and decoded output for the FIRST call (step 1 decode).
"""
import sys, torch

_VRAM_TOTAL = 12 * 1024**3
def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

from cache import (
    TurboQuantConfig, TurboQuantCache,
    turboquant_encode_internal as _orig_encode,
    turboquant_decode_single as _orig_decode,
    Codebook, RandomHadamardRotation,
    polarquant_encode, polarquant_decode, qjl_encode,
    TurboQuantCompressed,
)

# Monkey-patch encode to capture first call
_encode_call_count = [0]
_captured = {}

def patched_encode(x, codebook, rotation, S, mixed=None):
    _encode_call_count[0] += 1
    call_num = _encode_call_count[0]

    result = _orig_encode(x, codebook, rotation, S, mixed)

    # Only print on the 8th encode call — that's step 1 decode for layer 0 head 0
    # (7 prefill tokens → 7 encode calls, then step 1 decode adds 1)
    if call_num == 8:
        k_dec = _orig_decode(result)
        print(f"\n=== ENCODE CALL #{call_num} (step 1 decode, layer 0 head 0) ===", flush=True)
        print(f"  k_vec[:8] (raw input):  {x[:8].tolist()}", flush=True)
        print(f"  k_dec[:8] (decoded):    {k_dec[:8].tolist()}", flush=True)
        _captured['k_vec'] = x.detach().clone()
        _captured['k_dec'] = k_dec.detach().clone()
        _captured['call_num'] = call_num

    return result

import cache as cache_module
cache_module.turboquant_encode_internal = patched_encode

# Monkey-patch decode too for comparison
_decode_call_count = [0]
def patched_decode(c):
    _decode_call_count[0] += 1
    result = _orig_decode(c)

    call_num = _decode_call_count[0]
    if call_num == 1:
        print(f"\n=== DECODE CALL #{call_num} (prefill token 0) ===", flush=True)
        print(f"  decoded[:8]: {result[:8].tolist()}", flush=True)
        _captured['prefill_decoded'] = result.detach().clone()

    return result

cache_module.turboquant_decode_single = patched_decode

from transformers import AutoModelForCausalLM, AutoTokenizer
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
print(f"Hook installed. encode calls so far: {_encode_call_count[0]}")
print(f"Running 1 decode step...\n", flush=True)

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=1,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nOutput: {gen!r}")
print(f"\nTotal encode calls: {_encode_call_count[0]}")
print(f"Total decode calls: {_decode_call_count[0]}")

if 'k_vec' in _captured and 'prefill_decoded' in _captured:
    cos_sim = torch.nn.functional.cosine_similarity(
        _captured['k_dec'].flatten(), _captured['prefill_decoded'].flatten(), dim=0
    ).item()
    print(f"\nCosine similarity (step1 decoded K vs prefill token 0 decoded K): {cos_sim:.6f}")
    print(f"Match (atol=1e-3)? {torch.allclose(_captured['k_dec'], _captured['prefill_decoded'], atol=1e-3)}")

unhook(model)
