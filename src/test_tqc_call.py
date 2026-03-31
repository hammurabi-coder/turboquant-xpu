"""Debug turboquant_attention call."""
import sys, os, torch
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
try:
    torch.xpu.mem_get_info = lambda d=None: (int(12.5*1024**3) - torch.xpu.memory_allocated(), int(12.5*1024**3))
except: pass

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import turboquant_encode_internal, TurboQuantConfig, turboquant_decode_single, turboquant_attention
from transformers import AutoModelForCausalLM

device = torch.device("xpu")
m = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True
).to(device).eval()
attn = m.model.layers[0].self_attn

tqc_config = TurboQuantConfig(d=attn.head_dim)
k_raw = torch.randn(1, 8, 10, 128).float()  # [B, KV_heads, seq, D]
v_raw = torch.randn(1, 8, 10, 128).float()
rot = tqc_config.make_rotation(0, 0)
S = tqc_config.make_qjl_matrix(0, 0)

# Encode using the same shape as prefill: k_raw[:, kv_head, :, :] = [1, seq, D]
k_enc = turboquant_encode_internal(k_raw[:, 0, :, :].float(), tqc_config.codebook, rot, S, mixed=None)
v_enc = turboquant_encode_internal(v_raw[:, 0, :, :].float(), tqc_config.codebook, rot, S, mixed=None)

print(f"k_enc type: {type(k_enc)}")
print(f"k_enc.pq.indices shape: {k_enc.pq.indices.shape}")
print(f"k_enc.pq.centroids shape: {k_enc.pq.codebook.centroids.shape}")

ck = [[k_enc]]
cv = [[v_enc]]

# Debug: what does turboquant_attention actually access?
print(f"len(ck)={len(ck)}, len(ck[0])={len(ck[0])}")
print(f"ck[0][0] type: {type(ck[0][0])}")

q = torch.randn(1, 2, 1, 128)  # 2 Q heads
print(f"q.shape: {q.shape}")

# Try calling with layer_idx=0 (required positional)
try:
    out = turboquant_attention(q, ck, cv, tqc_config, layer_idx=0)
    print(f"out shape: {out.shape}")
    print("SUCCESS")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
