"""
Phase 1 — Download and cache model to local disk.
One-time cost: ~3-5 min download + load.
After this, subsequent runs use the local cache.
"""
import os
import torch

os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"

# ── Patch mem_get_info for Arc B580 ──────────────────────────────────────────
_VRAM_TOTAL = int(12.5 * 1024**3)
def _mem_get_info_patch(device=None):
    allocated = torch.xpu.memory_allocated()
    return (_VRAM_TOTAL - allocated, _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "Qwen/Qwen3-4B"
CACHE_DIR = "/home/hermes/.cache/huggingface/modules"

print(f"Downloading and caching: {MODEL_NAME}")
print(f"Cache dir: {CACHE_DIR}")

# ── download / cache tokenizer ───────────────────────────────────────────────
print("\n[1/2] Caching tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="left",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(CACHE_DIR)
print(f"  Tokenizer saved to {CACHE_DIR}")

# ── download / cache model weights ───────────────────────────────────────────
print("\n[2/2] Downloading and caching model weights...")
print("  (Loading to CPU first, then moving to XPU — avoids mem_get_info crash)")
reset = torch.xpu.memory_allocated()  # should be ~0
print(f"  VRAM before load: {torch.xpu.memory_allocated() / 1e9:.2f}GB")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.float16,
    device_map="cpu",         # load to CPU RAM — avoids XPU allocator warmup
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

torch.xpu.synchronize()
print(f"  VRAM after CPU load (still on CPU): {torch.xpu.memory_allocated() / 1e9:.2f}GB")

# Save in HF format (still quantized — BitsAndBytes saves in 4-bit)
save_path = os.path.join(CACHE_DIR, "models", "Qwen3-4B-4bit")
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
print(f"  Model saved to: {save_path}")

# Save safetensors index if missing
import glob
saved = glob.glob(os.path.join(save_path, "*.safetensors"))
print(f"  Safetensors files: {len(saved)}")

print("\n✓ Phase 1 complete. Run phase2_benchmark.py next.")
