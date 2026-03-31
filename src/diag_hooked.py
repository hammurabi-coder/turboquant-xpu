"""
Diagnostic Part 2: Hooked run — fresh model load, hook installed, measure.
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
MAX_NEW_TOKENS = 5

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

# Prefill (no hook yet — just to get pkv)
ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    pkv = model(ids, output_hidden_states=False).past_key_values
print(f"Prefill done, pkv tokens: {pkv.layers[0].keys.shape[2]}")

# Build config from actual head_dim
head_dim = model.model.layers[0].self_attn.head_dim
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

# Install hook (compresses pkv and replaces forwards)
tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Intercept the layer-0 hooked forward to measure its output
_hooked_fwd = model.model.layers[0].self_attn.forward
stats = []

def intercept(*args, **kwargs):
    out, _ = _hooked_fwd(*args, **kwargs)
    s = {"mean": out.mean().item(), "std": out.std().item(),
         "max_abs": out.abs().max().item(), "shape": tuple(out.shape)}
    stats.append(s)
    print(f"  [step {len(stats)}] mean={s['mean']:+.6f}  std={s['std']:.6f}  |max|={s['max_abs']:.6f}  shape={s['shape']}", flush=True)
    return out, None

model.model.layers[0].self_attn.forward = intercept

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nOutput: {gen!r}")

# Step-1 logits
unhook(model)
model.model.layers[0].self_attn.forward = model.model.layers[0].self_attn.forward  # restore original
# Reload to get clean step-1
model2 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map={"": "xpu"},
                                               torch_dtype=torch.float32, trust_remote_code=True)
model2.eval()
ids_slim = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    pkv2 = model2(ids_slim, output_hidden_states=False).past_key_values
    install_turboquant_hook(model2, tqc_config, pkv2)
    r = model2.generate(ids_slim, max_new_tokens=1, do_sample=False,
                          pad_token_id=tok.pad_token_id,
                          output_scores=True, return_dict_in_generate=True)
    lg = r.scores[0][0]
    v, idx = lg.topk(5)
    print("\nStep-1 top-5:")
    for val, tid in zip(v, idx):
        print(f"  {val.item():+.4f}  id={tid.item():>6}  {tok.decode([tid.item()])!r}")

unhook(model)
unhook(model2)

print("\n=== HOOKED_STATS ===")
for i, s in enumerate(stats):
    print(f"step {i+1}: mean={s['mean']:+.8f} std={s['std']:.8f} max_abs={s['max_abs']:.8f}")
print("=== HOOKED_STATS_END ===")
