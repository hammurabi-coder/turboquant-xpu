"""Debug: time each phase of the TurboQuant pipeline."""
import os
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"

import torch
import time
_VRAM_TOTAL = int(12.5 * 1024**3)
torch.xpu.mem_get_info = lambda d=None: (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)

import sys
SRC_DIR = "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src"
sys.path.insert(0, SRC_DIR)
from cache import TurboQuantConfig, N_OUTLIER_CHANNELS, polarquant_decode, turboquant_encode_internal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

device = torch.device("xpu")

tokenizer = AutoTokenizer.from_pretrained(
    "/home/hermes/.cache/huggingface/modules",
    trust_remote_code=True, padding_side="left"
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/hermes/.cache/huggingface/modules/models/Qwen3-4B-4bit",
    device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
)
model = model.to(device)
model.eval()
torch.xpu.synchronize()
print(f"Model loaded. VRAM: {torch.xpu.memory_allocated()/1e9:.2f}GB\n")

# Prompt
text = "In the northern reaches of the Ember Valley, the last remnants of the ancient Clockwork Empire lay buried beneath layers of volcanic ash and forgotten memory. For three centuries, the great brass automatons had stood frozen in their eternal gardens, gears seized by the cold that swept down from the Serac peaks after the Cataclysm of the Second Sun. Scholars from the Athenaeum of Dusk had catalogued over four thousand distinct automaton designs."
tokens = tokenizer.encode(text, add_special_tokens=False)[:800]
prompt = tokenizer.decode(tokens)
print(f"Prompt: {len(tokens)} tokens\n")

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Time: prefill
print("=== PREFILL ===")
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    prefill_out = model(input_ids=inputs["input_ids"], use_cache=True)
torch.xpu.synchronize()
t_prefill = time.time() - t0
print(f"Prefill time:       {t_prefill:.1f}s")
print(f"VRAM after prefill: {torch.xpu.memory_allocated()/1e9:.2f}GB")

# Extract KV — DynamicCache stores as pkv.layers[l].keys/values
pkv = prefill_out.past_key_values
batch = 1
n_layers = len(pkv.layers)
n_heads = pkv.layers[0].keys.shape[1]
seq_len = pkv.layers[0].keys.shape[2]
head_dim = pkv.layers[0].keys.shape[3]
print(f"KV shape: layers={n_layers}, heads={n_heads}, seq={seq_len}, dim={head_dim}")

# Time: config creation
t0 = time.time()
config = TurboQuantConfig(
    d=head_dim, b_mse=3, device=device,
    mixed_precision=True, n_outlier=N_OUTLIER_CHANNELS,
    b_outlier=4, use_online_codebook=True,
)
torch.xpu.synchronize()
t_config = time.time() - t0
print(f"\nConfig creation:    {t_config*1000:.1f}ms")

# Time: compression of ONE layer ONE head
layer_idx, head_idx = 0, 0
k = pkv.layers[layer_idx].keys[:, head_idx:head_idx+1, :, :]  # [batch, 1, seq, head_dim]
v = pkv.layers[layer_idx].values[:, head_idx:head_idx+1, :, :]

k_flat = k.reshape(-1, head_dim).float()
v_flat = v.reshape(-1, head_dim).float()

t0 = time.time()
rotation = config.make_rotation(layer_idx, head_idx)
S = config.make_qjl_matrix(layer_idx, head_idx)
mixed = config.get_mixed_config(layer_idx, head_idx, k_flat)
k_comp = turboquant_encode_internal(k_flat, config.codebook, rotation, S, mixed=mixed)
v_comp = turboquant_encode_internal(v_flat, config.codebook, rotation, S, mixed=mixed)
k_recon = polarquant_decode(k_comp.pq)[..., :head_dim]
v_recon = polarquant_decode(v_comp.pq)[..., :head_dim]
torch.xpu.synchronize()
t_one = time.time() - t0
print(f"Compress 1L×1H:    {t_one*1000:.1f}ms")

# Time: compress ALL layers and heads
t0 = time.time()
all_k_recon = []
all_v_recon = []
for l in range(n_layers):
    layer_k = []
    layer_v = []
    for h in range(n_heads):
        k_f = pkv.layers[l].keys[:, h:h+1, :, :].reshape(-1, head_dim).float()
        v_f = pkv.layers[l].values[:, h:h+1, :, :].reshape(-1, head_dim).float()
        rot = config.make_rotation(l, h)
        S_mat = config.make_qjl_matrix(l, h)
        mix = config.get_mixed_config(l, h, k_f)
        k_c = turboquant_encode_internal(k_f, config.codebook, rot, S_mat, mixed=mix)
        v_c = turboquant_encode_internal(v_f, config.codebook, rot, S_mat, mixed=mix)
        k_r = polarquant_decode(k_c.pq)[..., :head_dim]
        v_r = polarquant_decode(v_c.pq)[..., :head_dim]
        layer_k.append(k_r)
        layer_v.append(v_r)
    all_k_recon.append(torch.cat(layer_k, dim=1))  # [batch, n_heads, seq, head_dim]
    all_v_recon.append(torch.cat(layer_v, dim=1))
torch.xpu.synchronize()
t_full_compress = time.time() - t0
print(f"Full KV compress:   {t_full_compress:.1f}s")
print(f"  (= {t_full_compress*1000:.0f}ms for {n_layers}L × {n_heads}H)")

# Rebuild cache
t0 = time.time()
new_cache = DynamicCache()
for l in range(n_layers):
    new_cache.update(all_k_recon[l], all_v_recon[l], layer_idx=l)
torch.xpu.synchronize()
t_rebuild = time.time() - t0
print(f"Cache rebuild:       {t_rebuild*1000:.1f}ms")

# First token
with torch.no_grad():
    logits = prefill_out.logits[:, -1, :]
    next_tok = logits.argmax(dim=-1, keepdim=True)
    gen_ids = torch.cat([inputs["input_ids"], next_tok], dim=-1)

# Decode loop: 10 tokens
print(f"\n=== DECODE LOOP (10 tokens) ===")
t0 = time.time()
for i in range(10):
    with torch.no_grad():
        out = model(input_ids=next_tok, past_key_values=new_cache, use_cache=True)
        new_cache = out.past_key_values
        logits = out.logits[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
torch.xpu.synchronize()
t_decode = time.time() - t0
print(f"10-token decode:    {t_decode:.1f}s")
print(f"  (= {t_decode/10*1000:.0f}ms per token)")
print(f"VRAM after decode:  {torch.xpu.memory_allocated()/1e9:.2f}GB")

print(f"\nGenerated text:\n{tokenizer.decode(gen_ids[0])}")
print(f"\nTotal VRAM:        {torch.xpu.memory_allocated()/1e9:.2f}GB")
