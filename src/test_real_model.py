"""
TurboQuant real-model validation on Mistral-7B-Instruct-v0.3.

Runs two mixed-precision modes:
  - 2.5-bit headline mode: b_mse=2, b_outlier=3
  - 3.5-bit headline mode: b_mse=3, b_outlier=4

Both modes use:
  - calibrated original-space outlier selection
  - two independent rotations for regular vs outlier channels
  - online codebooks computed from the actual prompt KV distribution
"""

import io
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cache import (
    N_OUTLIER_CHANNELS,
    TurboQuantConfig,
    compression_ratio_fp16,
    memory_bytes_per_vector,
    polarquant_decode,
    turboquant_encode_internal,
)


MODE_SPECS = [
    {"name": "2.5-bit", "b_mse": 2, "b_outlier": 3},
    {"name": "3.5-bit", "b_mse": 3, "b_outlier": 4},
]

TEST_PROMPTS = [
    "The capital of France is",
    "In quantum physics, the uncertainty principle states that",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The key difference between TCP and UDP is",
    "Once upon a time in a land far away, there lived a",
]

GEN_PROMPTS = [
    "The meaning of life is",
    "def quicksort(arr):",
    "In 1969, humans first",
]


def build_config(head_dim: int, device: torch.device, mode: Dict[str, int]) -> TurboQuantConfig:
    return TurboQuantConfig(
        d=head_dim,
        b_mse=mode["b_mse"],
        device=device,
        mixed_precision=True,
        n_outlier=N_OUTLIER_CHANNELS,
        b_outlier=mode["b_outlier"],
        use_online_codebook=True,
    )


def extract_kv_tuple(past_kv) -> Tuple:
    if hasattr(past_kv, "key_cache"):
        return tuple((past_kv.key_cache[l], past_kv.value_cache[l]) for l in range(len(past_kv.key_cache)))
    if hasattr(past_kv, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_kv.layers)
    if hasattr(past_kv, "__iter__"):
        return tuple((item[0], item[1]) for item in past_kv)
    return past_kv


def rebuild_dynamic_cache(kv_tuple: Tuple):
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx, (k, v) in enumerate(kv_tuple):
        cache.update(k, v, layer_idx)
    return cache


def compress_decompress_kv(
    past_key_values: Tuple,
    config: TurboQuantConfig,
) -> Tuple:
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

            k_compressed = turboquant_encode_internal(
                k_flat, config.codebook, rotation, S, mixed=mixed
            )
            v_compressed = turboquant_encode_internal(
                v_flat, config.codebook, rotation, S, mixed=mixed
            )

            k_recon = polarquant_decode(k_compressed.pq)[..., :head_dim].contiguous()
            v_recon = polarquant_decode(v_compressed.pq)[..., :head_dim].contiguous()

            new_k[:, head_idx] = k_recon.reshape(batch, seq_len, head_dim).to(k.dtype)
            new_v[:, head_idx] = v_recon.reshape(batch, seq_len, head_dim).to(v.dtype)

        new_past.append((new_k, new_v))

    return tuple(new_past)


def kv_reconstruction_error(original: Tuple, reconstructed: Tuple) -> List[dict]:
    errors = []
    for layer_idx, (orig_layer, recon_layer) in enumerate(zip(original, reconstructed)):
        k_orig, v_orig = orig_layer[0], orig_layer[1]
        k_recon, v_recon = recon_layer[0], recon_layer[1]
        k_cos = F.cosine_similarity(
            k_orig.float().reshape(1, -1), k_recon.float().reshape(1, -1)
        ).item()
        v_cos = F.cosine_similarity(
            v_orig.float().reshape(1, -1), v_recon.float().reshape(1, -1)
        ).item()
        errors.append({"layer": layer_idx, "key_cosine": k_cos, "val_cosine": v_cos})
    return errors


def generate_and_compare(model, tokenizer, prompt: str, mode: Dict[str, int], max_new_tokens: int = 30):
    device = next(model.parameters()).device

    with torch.no_grad():
        probe = tokenizer("test", return_tensors="pt").to(device)
        probe_out = model(**probe, use_cache=True)
        head_dim = extract_kv_tuple(probe_out.past_key_values)[0][0].shape[-1]

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    with torch.no_grad():
        normal_out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )
    normal_text = tokenizer.decode(normal_out[0], skip_special_tokens=True)

    generated_ids = input_ids.clone()
    with torch.no_grad():
        outputs = model(input_ids=generated_ids, use_cache=True)
        raw_kv = extract_kv_tuple(outputs.past_key_values)
        config = build_config(head_dim, device, mode)
        compressed_kv = compress_decompress_kv(raw_kv, config)
        past_kv = rebuild_dynamic_cache(compressed_kv)

        logits = outputs.logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        for _ in range(max_new_tokens - 1):
            outputs = model(
                input_ids=generated_ids[:, -1:],
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = logits.argmax(dim=-1, keepdim=True)
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    tq_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return normal_text, tq_text


def evaluate_mode(model, tokenizer, device: torch.device, head_dim: int, mode: Dict[str, int]) -> Dict[str, object]:
    all_cosines: List[float] = []
    all_top1_match: List[bool] = []
    all_top5_overlap: List[float] = []
    compress_times_ms: List[float] = []

    print("\n" + "=" * 72)
    print(f"  MODE: {mode['name']}  (b_mse={mode['b_mse']}, b_outlier={mode['b_outlier']})")
    print("=" * 72)

    for i, prompt in enumerate(TEST_PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_normal = model(**inputs, use_cache=True)
            logits_normal = out_normal.logits[:, -1, :]
            raw_kv = extract_kv_tuple(out_normal.past_key_values)

            config = build_config(head_dim, device, mode)
            t0 = time.time()
            kv_compressed = compress_decompress_kv(raw_kv, config)
            compress_times_ms.append((time.time() - t0) * 1000)

            compressed_cache = rebuild_dynamic_cache(kv_compressed)
            last_token = inputs.input_ids[:, -1:]
            out_compressed = model(
                input_ids=last_token,
                past_key_values=compressed_cache,
                use_cache=False,
            )
            logits_compressed = out_compressed.logits[:, -1, :]

        cosine = F.cosine_similarity(
            logits_normal.float().reshape(1, -1),
            logits_compressed.float().reshape(1, -1),
        ).item()
        top1_n = logits_normal.argmax(dim=-1)
        top1_c = logits_compressed.argmax(dim=-1)
        top1_match = bool((top1_n == top1_c).item())
        top5_n = set(logits_normal.topk(5).indices[0].tolist())
        top5_c = set(logits_compressed.topk(5).indices[0].tolist())
        top5_overlap = len(top5_n & top5_c) / 5.0

        all_cosines.append(cosine)
        all_top1_match.append(top1_match)
        all_top5_overlap.append(top5_overlap)

        print(f"\n  Prompt {i+1}: \"{prompt[:55]}\"")
        print(f"    Logit cosine sim:  {cosine:.6f}")
        print(f"    Top-1 match:       {'yes' if top1_match else 'no'}")
        print(f"    Top-5 overlap:     {top5_overlap:.0%}")
        print(f"    Compress time:     {compress_times_ms[-1]:.1f} ms")

    inputs = tokenizer(TEST_PROMPTS[0], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        raw_kv = extract_kv_tuple(out.past_key_values)
        config = build_config(head_dim, device, mode)
        kv_recon = compress_decompress_kv(raw_kv, config)
    layer_errors = kv_reconstruction_error(raw_kv, kv_recon)

    print("\n  Per-layer cosine breakdown")
    print(f"  {'Layer':>5}  {'Key cosine':>12}  {'Value cosine':>12}")
    print(f"  {'-' * 5}  {'-' * 12}  {'-' * 12}")
    for error in layer_errors:
        print(f"  {error['layer']:>5}  {error['key_cosine']:>12.6f}  {error['val_cosine']:>12.6f}")

    gen_matches = 0
    for prompt in GEN_PROMPTS:
        normal_text, tq_text = generate_and_compare(model, tokenizer, prompt, mode)
        normal_cont = normal_text[len(prompt):].strip()
        tq_cont = tq_text[len(prompt):].strip()
        gen_matches += int(normal_cont == tq_cont)

    avg_cosine = sum(all_cosines) / len(all_cosines)
    avg_top5 = sum(all_top5_overlap) / len(all_top5_overlap)
    avg_key_cos = sum(item["key_cosine"] for item in layer_errors) / len(layer_errors)
    avg_val_cos = sum(item["val_cosine"] for item in layer_errors) / len(layer_errors)
    tq_bytes, fp16_bytes = memory_bytes_per_vector(
        head_dim,
        b_mse=mode["b_mse"],
        mixed_precision=True,
        n_outlier=N_OUTLIER_CHANNELS,
        b_outlier=mode["b_outlier"],
    )
    actual_ratio = fp16_bytes / tq_bytes
    headline_ratio = compression_ratio_fp16(
        head_dim,
        b_mse=mode["b_mse"],
        mixed_precision=True,
        n_outlier=N_OUTLIER_CHANNELS,
        b_outlier=mode["b_outlier"],
    )

    result = {
        "mode": mode["name"],
        "avg_cosine": avg_cosine,
        "top1_rate": sum(all_top1_match) / len(all_top1_match),
        "avg_top5": avg_top5,
        "avg_key_cos": avg_key_cos,
        "avg_val_cos": avg_val_cos,
        "layer_errors": layer_errors,
        "gen_matches": gen_matches,
        "gen_total": len(GEN_PROMPTS),
        "headline_ratio": headline_ratio,
        "actual_ratio": actual_ratio,
        "tq_bytes": tq_bytes,
        "fp16_bytes": fp16_bytes,
        "avg_compress_ms": sum(compress_times_ms) / len(compress_times_ms),
    }

    print("\n  Summary")
    print(f"    Logit cosine similarity:  {avg_cosine:.6f}")
    print(f"    Top-1 prediction match:   {result['top1_rate']:.0%}")
    print(f"    Top-5 overlap:            {avg_top5:.0%}")
    print(f"    KV key cosine:            {avg_key_cos:.6f}")
    print(f"    KV value cosine:          {avg_val_cos:.6f}")
    print(f"    Generation match:         {gen_matches}/{len(GEN_PROMPTS)}")
    print(f"    Headline compression:     {headline_ratio:.2f}x vs FP16")
    print(f"    Actual bytes/vector:      {tq_bytes} vs {fp16_bytes} FP16 ({actual_ratio:.2f}x)")
    print(f"    Avg compress time:        {result['avg_compress_ms']:.1f} ms")

    return result


def main() -> bool:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 72)
    print("    TurboQuant -- Real Transformer Model Validation")
    print("=" * 72)

    model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    print(f"\n[*] Loading model: {model_name} (4-bit quantized)")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"   Using GPU: {torch.cuda.get_device_name()}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        device = torch.device("cpu")
        print("   Using CPU (no CUDA detected; expect a long run)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        model = model.to(device)
    model.eval()

    with torch.no_grad():
        test_input = tokenizer("test", return_tensors="pt").to(device)
        test_out = model(**test_input, use_cache=True)
        test_kv = extract_kv_tuple(test_out.past_key_values)
        head_dim = test_kv[0][0].shape[-1]
        del test_out, test_input

    print(f"   Head dim: {head_dim}")
    print(f"   Outlier channels: {N_OUTLIER_CHANNELS}")
    print("   Calibration: original-space variance + online codebook")

    results = [evaluate_mode(model, tokenizer, device, head_dim, mode) for mode in MODE_SPECS]

    print("\n" + "=" * 72)
    print("                         SUMMARY")
    print("=" * 72)
    for result in results:
        print(
            f"  {result['mode']:>7}: cosine={result['avg_cosine']:.6f}, "
            f"top1={result['top1_rate']:.0%}, key={result['avg_key_cos']:.6f}, "
            f"value={result['avg_val_cos']:.6f}, compression={result['headline_ratio']:.2f}x"
        )
    print("=" * 72)

    best = max(results, key=lambda item: item["avg_cosine"])
    return best["avg_cosine"] > 0.98


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
