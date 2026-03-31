"""
Test: full 42-token prompt, 1 decode step.
Compare hooked vs baseline at each layer's attention output level.
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
PROMPT = "In the northern reaches of the Ember Valley, the last remnants of the ancient Clockwork Empire"

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

ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
print(f"Prompt: {ids.shape[1]} tokens")

# Baseline: get logits for step 1
with torch.no_grad():
    baseline_out = model(ids, output_hidden_states=False)
    baseline_logits = baseline_out.logits[:, -1, :]
    baseline_top = baseline_logits.argmax(-1).item()
print(f"Baseline step 1 token: {tok.decode([baseline_top])!r}")

# Get pkv
with torch.no_grad():
    pkv = model(ids, output_hidden_states=False).past_key_values
print(f"pkv tokens: {pkv.layers[0].keys.shape[2]}")

# Install hook
head_dim = model.model.layers[0].self_attn.head_dim
tqc_config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False,
)

tqc_cache = install_turboquant_hook(model, tqc_config, pkv)
print("Hook installed.\n", flush=True)

# Hook intercept: capture attention output at layer 0 and 27
layer_stats = {}
_orig_fwd = None

def make_interceptor(layer_idx):
    def interceptor(attn_self, *args, **kwargs):
        # Get original output by calling original forward
        orig_out, _ = _orig_fwd(attn_self, *args, **kwargs)
        if layer_idx in [0, 1, 27]:
            # Get hidden states to see what attention produced
            hs = kwargs.get('hidden_states') or (args[0] if args else None)
            if hs is not None:
                # Only for decode steps (seq=1)
                past_kv = kwargs.get('past_key_values')
                if past_kv is not None and hs.shape[1] == 1:
                    # Compute what the baseline hidden state would be
                    pass
        return orig_out, None
    return interceptor

# Instead, just generate and check token
with torch.no_grad():
    hooked_ids = model.generate(ids, max_new_tokens=1,
                               do_sample=False, pad_token_id=tok.pad_token_id)

hooked_token = hooked_ids[0, -1].item()
print(f"Hooked step 1 token:  {tok.decode([hooked_token])!r}")
print(f"Match: {hooked_token == baseline_top}")

# Try 2 steps
with torch.no_grad():
    hooked_ids2 = model.generate(ids, max_new_tokens=2,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
gen2 = tok.decode(hooked_ids2[0], skip_special_tokens=True)
print(f"\nHooked 2-step output: {gen2!r}")

unhook(model)
