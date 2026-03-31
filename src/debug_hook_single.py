"""
Minimal single-step debug of the TurboQuant hook.
1. Prefill with model
2. Capture pkv
3. Install hook
4. Generate ONE token with hook active
5. Print attention output statistics from layer 0
6. Compare to what baseline attention would produce
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
PROMPT = "In the northern reaches of"

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

# Baseline: run prefill + 1 decode step with original attention
with torch.no_grad():
    baseline_out = model(ids, output_hidden_states=False)
    baseline_logits = baseline_out.logits[:, -1, :]
    baseline_top_token = baseline_logits.argmax(dim=-1).item()
print(f"Baseline prefill done. Top token at step 1: {tok.decode([baseline_top_token])!r}")
print(f"Baseline logits mean: {baseline_logits.mean():.4f}, std: {baseline_logits.std():.4f}")

# Get pkv from baseline
with torch.no_grad():
    pkv = model(ids, output_hidden_states=False).past_key_values
print(f"pkv tokens: {pkv.layers[0].keys.shape[2]}")

# Install hook
head_dim = model.model.layers[0].self_attn.head_dim
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

print("\nInstalling hook...")
tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Generate ONE token
with torch.no_grad():
    hooked_out_ids = model.generate(ids, max_new_tokens=1,
                                     do_sample=False, pad_token_id=tok.pad_token_id)

hooked_token = hooked_out_ids[0, -1].item()
hooked_text = tok.decode([hooked_token])
print(f"Hooked decode token 1: {hooked_text!r}")
print(f"Expected (baseline):   {tok.decode([baseline_top_token])!r}")
print(f"Match: {hooked_token == baseline_top_token}")

unhook(model)
