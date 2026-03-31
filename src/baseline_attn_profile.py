"""
Profile baseline attention output norms per layer, per step.
Then compare with hooked model.
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
PROMPT = "In the northern reaches of the Ember Valley"

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

# Get baseline norms per layer per step
n_layers = model.config.num_hidden_layers
n_heads = model.config.num_attention_heads
head_dim = model.model.layers[0].self_attn.head_dim

# Hook each layer to capture attention output norms
_orig_fwds = []
hooks_registered = []

def make_hook_capture(layer_idx):
    orig = _orig_fwds[layer_idx]
    def capture_hook(attn_self, *args, **kwargs):
        out, _ = orig(attn_self, *args, **kwargs)
        past_kv = kwargs.get('past_key_values')
        hs = kwargs.get('hidden_states') or (args[0] if args else None)
        if past_kv is not None and hs is not None and hs.shape[1] == 1:
            import sys
            print(f"[BASELINE layer={layer_idx}] attn_out norm={out.norm().item():.4f}", file=sys.stderr)
        return out, None
    return capture_hook

_orig_fwds = [layer.self_attn.forward for layer in model.model.layers]
for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_hook_capture(li)

print(f"\n=== BASELINE generate(1 token) ===", flush=True)
with torch.no_grad():
    baseline_ids = model.generate(ids, max_new_tokens=1,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
baseline_text = tok.decode(baseline_ids[0], skip_special_tokens=True)
print(f"Baseline 1-step: {baseline_text!r}")

# Restore
for li, layer in enumerate(model.model.layers):
    layer.self_attn.forward = _orig_fwds[li]

# Now same for hooked model
model2 = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    device_map={"": "xpu"},
    torch_dtype=torch.float32,
    trust_remote_code=True,
)
model2.eval()

with torch.no_grad():
    pkv = model2(ids, output_hidden_states=False).past_key_values

tqc_config = TurboQuantConfig(d=head_dim, b_mse=3, device=device,
                               mixed_precision=False, use_online_codebook=False)
tqc_cache = install_turboquant_hook(model2, tqc_config, pkv)

print(f"\n=== HOOKED generate(1 token) ===", flush=True)
with torch.no_grad():
    hooked_ids = model2.generate(ids, max_new_tokens=1,
                                  do_sample=False, pad_token_id=tok.pad_token_id)
hooked_text = tok.decode(hooked_ids[0], skip_special_tokens=True)
print(f"Hooked 1-step: {hooked_text!r}")

unhook(model2)
