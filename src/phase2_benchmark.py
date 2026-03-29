"""
Phase 2 — Run benchmarks against the locally-cached model.
Uses: /home/hermes/.cache/huggingface/modules/
"""
import gc
import os
import sys
import time
from typing import Tuple

os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"

import torch

# ── Patch mem_get_info for Arc B580 ──────────────────────────────────────────
_VRAM_TOTAL = int(12.5 * 1024**3)
def _mem_get_info_patch(device=None):
    allocated = torch.xpu.memory_allocated()
    return (_VRAM_TOTAL - allocated, _VRAM_TOTAL)
torch.xpu.mem_get_info = _mem_get_info_patch

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from cache import (
    TurboQuantConfig,
    N_OUTLIER_CHANNELS,
    polarquant_decode,
    turboquant_encode_internal,
)

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ── helpers from test_real_model.py ──────────────────────────────────────────

def extract_kv_tuple(past_kv):
    """Extract (k, v) tuples from any HuggingFace cache format.

    NOTE: DynamicCache is a generator — it yields items once and is consumed.
    Always convert to list() immediately to allow multiple passes.
    """
    if hasattr(past_kv, "layers"):
        # DynamicCache — use .layers attribute (works after exhaustion)
        return tuple((layer.keys, layer.values) for layer in past_kv.layers)
    if hasattr(past_kv, "key_cache"):
        # StaticCache or similar
        return tuple((past_kv.key_cache[l], past_kv.value_cache[l])
                     for l in range(len(past_kv.key_cache)))
    # Generator or iterable — consume immediately
    return tuple((item[0], item[1]) for item in past_kv)


def rebuild_dynamic_cache(kv_tuple: Tuple):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_tuple):
        cache.update(k, v, layer_idx)
    return cache


def compress_decompress_kv(past_key_values, config: TurboQuantConfig) -> Tuple:
    """Compress then decompress full KV cache using TurboQuant (3.5-bit mode).

    past_key_values: tuple of (k, v) from extract_kv_tuple(), where each is
    a [batch, n_heads, seq_len, head_dim] tensor.
    """
    new_past = []
    for layer_idx, layer_data in enumerate(past_key_values):
        k, v = layer_data[0], layer_data[1]
        batch, n_heads, seq_len, head_dim = k.shape
        new_k = torch.zeros_like(k)
        new_v = torch.zeros_like(v)
        for head_idx in range(n_heads):
            rotation = config.make_rotation(layer_idx, head_idx)
            S = config.make_qjl_matrix(layer_idx, head_idx)
            k_flat = k[:, head_idx, :, :].reshape(-1, head_dim).float().to(config.device)
            v_flat = v[:, head_idx, :, :].reshape(-1, head_dim).float().to(config.device)
            mixed = config.get_mixed_config(layer_idx, head_idx, k_flat)
            k_compressed = turboquant_encode_internal(k_flat, config.codebook, rotation, S, mixed=mixed)
            v_compressed = turboquant_encode_internal(v_flat, config.codebook, rotation, S, mixed=mixed)
            k_recon = polarquant_decode(k_compressed.pq)[..., :head_dim].contiguous()
            v_recon = polarquant_decode(v_compressed.pq)[..., :head_dim].contiguous()
            new_k[:, head_idx] = k_recon.reshape(batch, seq_len, head_dim).to(k.dtype)
            new_v[:, head_idx] = v_recon.reshape(batch, seq_len, head_dim).to(v.dtype)
        new_past.append((new_k, new_v))
    return tuple(new_past)


# ── VRAM helpers ──────────────────────────────────────────────────────────────

def vram_gb() -> float:
    torch.xpu.synchronize()
    return torch.xpu.memory_allocated() / 1e9

def peak_vram_gb() -> float:
    torch.xpu.synchronize()
    return torch.xpu.max_memory_allocated() / 1e9

def reset_vram():
    gc.collect()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    torch.xpu.reset_peak_memory_stats()

def build_tq_config(head_dim: int, device: torch.device, b_mse: int, b_outlier: int):
    return TurboQuantConfig(
        d=head_dim, b_mse=b_mse, device=device,
        mixed_precision=True, n_outlier=N_OUTLIER_CHANNELS,
        b_outlier=b_outlier, use_online_codebook=True,
    )


# ── prompt ────────────────────────────────────────────────────────────────────

def make_prompt(tokenizer, target_tokens: int = 800) -> str:
    paragraphs = [
        "In the northern reaches of the Ember Valley, the last remnants of the ancient "
        "Clockwork Empire lay buried beneath layers of volcanic ash and forgotten memory. "
        "For three centuries, the great brass automatons had stood frozen in their eternal "
        "gardens, gears seized by the cold that swept down from the Serac peaks after the "
        "Cataclysm of the Second Sun. Scholars from the Athenaeum of Dusk had catalogued "
        "over four thousand distinct automaton designs, ranging from the towering War Golems "
        "of the pre-Cataclysm era to the delicate Songbirds that once filled the palace "
        "corridors with artificial melody. None of them moved. None of them would ever move "
        "again — or so the textbooks claimed, until a young researcher named Dr. Mira Voss "
        "stumbled upon a faint electromagnetic pulse emanating from one of the oldest machines "
        "in the collection, suggesting that the great engines of the Empire had not truly died "
        "but merely been waiting, dormant and patient, for conditions that might one day allow "
        "them to wake once more and resume their ancient work.",
        "The discovery sent shockwaves through the academic community. For decades, the dominant "
        "theory held that the Clockwork Empire had collapsed due to a catastrophic thermal event "
        "that flash-froze its mechanical infrastructure beyond any possibility of repair. The "
        "volcanic winter that followed the Cataclysm had, according to this view, reduced even "
        "the most sophisticated machines to inert relics — beautiful but fundamentally broken "
        "artifacts of a civilization that had overreached the boundaries of what their technology "
        "could sustain. Dr. Voss's findings suggested something far more remarkable: that the "
        "machines had been designed with a form of deep-sleep resilience that allowed them to "
        "enter a state of suspended animation precisely calibrated to survive exactly these "
        "kinds of planetary-scale thermal disruptions. The pulse she detected was not a "
        "malfunction or an artifact of her instruments. It was a heartbeat, slow and steady, "
        "maintained across three hundred years of continuous dormancy by mechanisms that "
        "engineers today would struggle to replicate with anything approaching comparable "
        "efficiency or longevity.",
        "The implications were staggering. If the automatons of the Empire had truly survived "
        "in a dormant state, then the Serac peaks and the surrounding Ember Valley might contain "
        "thousands of machines in various states of preservation — some perhaps still "
        "functional, others merely awaiting the right conditions to be restored to active "
        "service. The strategic, economic, and cultural significance of such a discovery "
        "defied any straightforward estimation. Within weeks, the site of Voss's original "
        "finding had been transformed into the most heavily funded archaeological excavation "
        "in the history of the known world, with representatives from seventeen nations "
        "competing for access to the findings and the honor of participating in what many "
        "were already calling the most important discovery since the unearthing of the "
        "Original Libraries beneath the dried seabed of the Former Ocean.",
    ]
    text = " ".join(paragraphs)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


# ── baseline pass ─────────────────────────────────────────────────────────────

def run_baseline(model, tokenizer, prompt: str, max_new_tokens: int = 100):
    reset_vram()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    peak_vram = peak_vram_gb()
    generated = output_ids.shape[1] - prompt_len
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, peak_vram, generated, elapsed


# ── turboquant pass ────────────────────────────────────────────────────────────

def run_turboquant(model, tokenizer, prompt: str, b_mse=3, b_outlier=4, max_new_tokens=100):
    reset_vram()
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    prompt_len = input_ids.shape[1]

    # probe head_dim
    with torch.no_grad():
        probe = tokenizer("probe", return_tensors="pt").to(device)
        probe_out = model(**probe, use_cache=True)
        head_dim = extract_kv_tuple(probe_out.past_key_values)[0][0].shape[-1]
        del probe_out, probe

    # prefill
    t_prefill = time.time()
    with torch.no_grad():
        prefill_out = model(input_ids=input_ids, use_cache=True)
    prefill_sec = time.time() - t_prefill

    raw_kv = extract_kv_tuple(prefill_out.past_key_values)

    # compress
    t_compress = time.time()
    config = build_tq_config(head_dim, device, b_mse, b_outlier)
    compressed_kv = compress_decompress_kv(raw_kv, config)
    compress_ms = (time.time() - t_compress) * 1000

    past_kv = rebuild_dynamic_cache(compressed_kv)

    # first token
    with torch.no_grad():
        logits = prefill_out.logits[:, -1, :]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([input_ids, next_tok], dim=-1)

    # decode loop
    t_decode = time.time()
    for _ in range(max_new_tokens - 1):
        with torch.no_grad():
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            next_tok = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_tok], dim=-1)
            if next_tok.item() == tokenizer.eos_token_id:
                break
    decode_sec = time.time() - t_decode

    peak_vram = peak_vram_gb()
    generated = generated_ids.shape[1] - prompt_len
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return {
        "text": text,
        "generated_tokens": generated,
        "peak_vram": peak_vram,
        "compress_ms": compress_ms,
        "prefill_sec": prefill_sec,
        "decode_sec": decode_sec,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  TurboQuant — Qwen3-4B on Intel Arc B580 (XPU)")
    print("  Phase 2: Local-cache benchmark")
    print("=" * 72)

    device = torch.device("xpu")
    print(f"\nDevice: {torch.xpu.get_device_name(0)}")
    print(f"VRAM total: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # ── load from local cache ─────────────────────────────────────────────────
    CACHE_DIR = "/home/hermes/.cache/huggingface/modules"
    MODEL_PATH = os.path.join(CACHE_DIR, "models", "Qwen3-4B-4bit")

    if not os.path.exists(MODEL_PATH):
        print(f"\nERROR: Model not found at {MODEL_PATH}")
        print("  Run phase1_save_model.py first to download and cache the model.")
        return

    print(f"\nLoading model from local cache: {MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(
        CACHE_DIR,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    reset_vram()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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
    model = model.to(device)
    model.eval()

    print(f"  Model loaded. VRAM: {vram_gb():.2f}GB")

    # ── build prompt ──────────────────────────────────────────────────────────
    print("\nBuilding ~800-token prompt...")
    prompt = make_prompt(tokenizer, target_tokens=800)
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"  Prompt tokens: {prompt_tokens}")

    # ── pass A: baseline ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PASS A — BASELINE (standard HuggingFace generate)")
    print("=" * 72)
    baseline_text, baseline_vram, baseline_new, baseline_time = run_baseline(
        model, tokenizer, prompt, max_new_tokens=10
    )
    print(f"\n  Peak VRAM:      {baseline_vram:.2f}GB")
    print(f"  Tokens gen:     {baseline_new}")
    print(f"  Total time:    {baseline_time:.1f}s")
    print(f"  Output[0:300]: {baseline_text[:300]}")

    # ── pass B: turboquant ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  PASS B — TURBOQUANT (3.5-bit, online codebook)")
    print("=" * 72)
    tq = run_turboquant(model, tokenizer, prompt, b_mse=3, b_outlier=4, max_new_tokens=10)
    print(f"\n  Prefill time:   {tq['prefill_sec']:.1f}s")
    print(f"  Compress time:  {tq['compress_ms']:.1f}ms")
    print(f"  Decode time:    {tq['decode_sec']:.1f}s")
    print(f"  Peak VRAM:      {tq['peak_vram']:.2f}GB")
    print(f"  Tokens gen:     {tq['generated_tokens']}")
    print(f"  Output[0:300]: {tq['text'][:300]}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  RESULTS SUMMARY")
    print("=" * 72)
    print(f"\n  Model:          Qwen/Qwen3-4B-4bit (local cache)")
    print(f"  Context tokens: {prompt_tokens}")
    print(f"  Generation:     10 new tokens")
    print(f"\n  {'Pass':<20} {'Peak VRAM':<12} {'Total time':<12} {'VRAM delta'}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Baseline':<20} {baseline_vram:<12.2f} {baseline_time:<12.1f} {'(reference)':<12}")
    delta = tq['peak_vram'] - baseline_vram
    print(f"  {'TurboQuant 3.5b':<20} {tq['peak_vram']:<12.2f} "
          f"{tq['prefill_sec']+tq['compress_ms']/1000+tq['decode_sec']:<12.1f} "
          f"{'+' if delta > 0 else ''}{delta:.2f}GB")
    print(f"\n  Memory delta:     {'+' if delta > 0 else ''}{delta:.2f}GB  "
          f"({'BETTER' if delta < 0 else 'WORSE' if delta > 0 else 'SAME'})")
    print(f"  Compression overhead: {tq['compress_ms']:.1f}ms")

    # coherence check
    keywords = ["aeon-1142", "voss", "automaton", "clockwork", "ember", "serac", "pulse"]
    bl = baseline_text.lower()
    tl = tq["text"].lower()
    print(f"\n  Coherence check:")
    all_ok = True
    for kw in keywords:
        b = kw in bl
        t = kw in tl
        status = "OK" if (b and t) else "WARN"
        if not (b and t):
            all_ok = False
        print(f"    [{status}] '{kw}': baseline={b}, turboquant={t}")

    if all_ok:
        print("\n  Both outputs coherent — key entities present in both.")
    else:
        print("\n  WARNING: Some key entities missing.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
