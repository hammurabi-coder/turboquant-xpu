"""Minimal debug: time just prefill + extract + compress + 10-token decode."""
import os, sys
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
_VRAM_TOTAL = int(12.5 * 1024**3)
torch.xpu.mem_get_info = lambda d=None: (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)
import time

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import TurboQuantConfig, N_OUTLIER_CHANNELS, polarquant_decode, turboquant_encode_internal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

device = torch.device("xpu")
SRC = "/home/hermes/.cache/huggingface/modules"

print("Loading tokenizer..."); tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token

print("Loading model..."); m = AutoModelForCausalLM.from_pretrained(f"{SRC}/models/Qwen3-4B-4bit", device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
m = m.to(device); m.eval(); torch.xpu.synchronize()
print(f"Model on XPU. VRAM={torch.xpu.memory_allocated()/1e9:.2f}GB")

text = "In the northern reaches of the Ember Valley, the last remnants of the ancient Clockwork Empire lay buried. Scholars had catalogued over four thousand distinct automaton designs."
inp = {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()}

# Prefill
print("\n=== PREFILL ===")
torch.xpu.synchronize(); t0 = time.time()
with torch.no_grad(): out = m(input_ids=inp["input_ids"], use_cache=True)
torch.xpu.synchronize(); print(f"Prefill: {time.time()-t0:.1f}s  VRAM={torch.xpu.memory_allocated()/1e9:.2f}GB")

# Extract KV
pkv = out.past_key_values
print(f"pkv type: {type(pkv)}, len={len(pkv.layers)}")
layer0 = pkv.layers[0]
print(f"L0 keys shape: {layer0.keys.shape}, device: {layer0.keys.device}")

# Compress: 1 layer, 1 head
print("\n=== COMPRESS 1L x 1H ===")
cfg = TurboQuantConfig(d=128, b_mse=3, device=device, mixed_precision=True, n_outlier=32, b_outlier=4, use_online_codebook=True)
k0 = pkv.layers[0].keys[:, 0:1, :, :]  # [1, 1, seq, 128]
v0 = pkv.layers[0].values[:, 0:1, :, :]
k0_f = k0.reshape(-1, 128).float()
v0_f = v0.reshape(-1, 128).float()
torch.xpu.synchronize(); t0 = time.time()
rot = cfg.make_rotation(0, 0); S = cfg.make_qjl_matrix(0, 0)
mix = cfg.get_mixed_config(0, 0, k0_f)
kc = turboquant_encode_internal(k0_f, cfg.codebook, rot, S, mixed=mix)
vc = turboquant_encode_internal(v0_f, cfg.codebook, rot, S, mixed=mix)
kr = polarquant_decode(kc.pq)[..., :128]
vr = polarquant_decode(vc.pq)[..., :128]
torch.xpu.synchronize(); print(f"1L1H compress: {(time.time()-t0)*1000:.0f}ms")

# Full compress: 36 layers x 8 heads
print("\n=== FULL COMPRESS (36L x 8H) ===")
torch.xpu.synchronize(); t0 = time.time()
rebuilt = []
for l in range(len(pkv.layers)):
    new_k_parts, new_v_parts = [], []
    for h in range(pkv.layers[l].keys.shape[1]):
        k_f = pkv.layers[l].keys[:, h:h+1, :, :].reshape(-1, 128).float()
        v_f = pkv.layers[l].values[:, h:h+1, :, :].reshape(-1, 128).float()
        rot = cfg.make_rotation(l, h); S = cfg.make_qjl_matrix(l, h)
        mix = cfg.get_mixed_config(l, h, k_f)
        kc = turboquant_encode_internal(k_f, cfg.codebook, rot, S, mixed=mix)
        vc = turboquant_encode_internal(v_f, cfg.codebook, rot, S, mixed=mix)
        kr = polarquant_decode(kc.pq)[..., :128].contiguous()
        vr = polarquant_decode(vc.pq)[..., :128].contiguous()
        new_k_parts.append(kr); new_v_parts.append(vr)
    new_k = torch.cat(new_k_parts, dim=1)  # [1, 8, seq, 128]
    new_v = torch.cat(new_v_parts, dim=1)
    rebuilt.append((new_k.to(device), new_v.to(device)))
torch.xpu.synchronize(); compress_time = time.time() - t0
print(f"Full compress: {compress_time:.1f}s  VRAM={torch.xpu.memory_allocated()/1e9:.2f}GB")

# Rebuild cache
print("\n=== REBUILD CACHE ===")
t0 = time.time()
cache = DynamicCache()
for l in range(len(rebuilt)):
    cache.update(rebuilt[l][0], rebuilt[l][1], layer_idx=l)
print(f"Rebuild: {time.time()-t0:.1f}s")

# First token
logits = out.logits[:, -1, :]
next_tok = logits.argmax(dim=-1, keepdim=True)
gen_ids = torch.cat([inp["input_ids"], next_tok], dim=-1)

# 10-token decode
print("\n=== 10-TOKEN DECODE ===")
torch.xpu.synchronize(); t0 = time.time()
for i in range(10):
    with torch.no_grad():
        o = m(input_ids=next_tok, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        next_tok = o.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
torch.xpu.synchronize(); t_decode = time.time() - t0
print(f"10-token decode: {t_decode:.1f}s ({(t_decode/10)*1000:.0f}ms/token)")
print(f"VRAM after decode: {torch.xpu.memory_allocated()/1e9:.2f}GB")
print(f"\nOutput: {tok.decode(gen_ids[0])}")
print("\nAll steps completed successfully!")
