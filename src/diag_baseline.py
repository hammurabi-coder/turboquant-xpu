"""
Diagnostic Part 1: Baseline (no hook) — fresh model load.
"""
import sys, torch

_VRAM_TOTAL = 12 * 1024**3
def _mem_get_info_patch(d=None):
    return (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")

from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Intercept layer 0
_orig = model.model.layers[0].self_attn.forward
stats = []

def intercept(*args, **kwargs):
    out, _ = _orig(*args, **kwargs)
    s = {"mean": out.mean().item(), "std": out.std().item(),
         "max_abs": out.abs().max().item(), "shape": tuple(out.shape)}
    stats.append(s)
    print(f"  [step {len(stats)}] mean={s['mean']:+.6f}  std={s['std']:.6f}  |max|={s['max_abs']:.6f}  shape={s['shape']}", flush=True)
    return out, None

model.model.layers[0].self_attn.forward = intercept

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
print(f"Input: {tok.decode(ids[0])!r}\n")

with torch.no_grad():
    out_ids = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS,
                              do_sample=False, pad_token_id=tok.pad_token_id)
gen = tok.decode(out_ids[0], skip_special_tokens=True)
print(f"\nOutput: {gen!r}")

# Step-1 logits
model.model.layers[0].self_attn.forward = _orig
ids_slim = tok(PROMPT, return_tensors="pt").input_ids.to(device)
with torch.no_grad():
    r = model.generate(ids_slim, max_new_tokens=1, do_sample=False,
                         pad_token_id=tok.pad_token_id,
                         output_scores=True, return_dict_in_generate=True)
    lg = r.scores[0][0]
    v, idx = lg.topk(5)
    print("\nStep-1 top-5:")
    for val, tid in zip(v, idx):
        print(f"  {val.item():+.4f}  id={tid.item():>6}  {tok.decode([tid.item()])!r}")

print("\n=== BASELINE_STATS ===")
for i, s in enumerate(stats):
    print(f"step {i+1}: mean={s['mean']:+.8f} std={s['std']:.8f} max_abs={s['max_abs']:.8f}")
print("=== BASELINE_STATS_END ===")
