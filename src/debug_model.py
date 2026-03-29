"""Debug: simplest possible model load + forward pass on XPU."""
import os
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"

import torch
_VRAM_TOTAL = int(12.5 * 1024**3)
torch.xpu.mem_get_info = lambda d=None: (_VRAM_TOTAL - torch.xpu.memory_allocated(), _VRAM_TOTAL)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CACHE_DIR = "/home/hermes/.cache/huggingface/modules"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CACHE_DIR, trust_remote_code=True, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model from disk cache...")
reset = torch.xpu.memory_allocated()
print(f"VRAM before: {reset / 1e9:.2f}GB")

model = AutoModelForCausalLM.from_pretrained(
    os.path.join(CACHE_DIR, "models", "Qwen3-4B-4bit"),
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"VRAM after CPU load: {torch.xpu.memory_allocated() / 1e9:.2f}GB")

print("Moving to XPU...")
model = model.to(torch.device("xpu"))
model.eval()
torch.xpu.synchronize()
print(f"VRAM after XPU move: {torch.xpu.memory_allocated() / 1e9:.2f}GB")

print("\nRunning single forward pass...")
inputs = tokenizer("Hello world", return_tensors="pt")
inputs = {k: v.to(torch.device("xpu")) for k, v in inputs.items()}
torch.xpu.synchronize()
import time
t0 = time.time()
with torch.no_grad():
    output = model(**inputs)
torch.xpu.synchronize()
t1 = time.time()
print(f"Forward pass took: {t1-t0:.2f}s")
print(f"Output logits shape: {output.logits.shape}")
print(f"VRAM after forward: {torch.xpu.memory_allocated() / 1e9:.2f}GB")

print("\nRunning 10-token generation (no cache)...")
input_ids = inputs["input_ids"]
torch.xpu.synchronize()
t0 = time.time()
with torch.no_grad():
    gen = model.generate(input_ids, max_new_tokens=10, do_sample=False)
torch.xpu.synchronize()
t1 = time.time()
print(f"10-token generate took: {t1-t0:.2f}s")
print(f"Generated: {tokenizer.decode(gen[0])}")
print(f"VRAM: {torch.xpu.memory_allocated() / 1e9:.2f}GB")

print("\nAll good!")
