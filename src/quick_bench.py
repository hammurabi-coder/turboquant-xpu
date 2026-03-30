"""Quick end-to-end: prefill + TurboQuant compress + 10-token decode.

Fixed memory measurement:
  - Baseline: measure peak during standard prefill+decode
  - TurboQuant: measure baseline prefill KV size, then compress, then
    del original KV, sync, measure compressed footprint BEFORE rebuilding cache.
    This gives the true compressed KV size without holding original + decompressed
    copies simultaneously.
"""
import os, sys, time
os.environ["BNB_CUDA_TRITON"] = "0"
os.environ["PYTORCH_XPU_ALLOC_CONF"] = "expandable_segments:True"
import torch
_V = int(12.5 * 1024**3)
torch.xpu.mem_get_info = lambda d=None: (_V - torch.xpu.memory_allocated(), _V)

sys.path.insert(0, "/home/hermes/turboquant-experiments/OnlyTerp-turboquant/src")
from cache import TurboQuantConfig, N_OUTLIER_CHANNELS, polarquant_decode, turboquant_encode_internal
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DynamicCache

device = torch.device("xpu")

# ── helpers ───────────────────────────────────────────────────────────────────
def kv_bytes(pkv):
    """Return bytes used by a DynamicCache's KV tensors."""
    total = 0
    for layer in pkv.layers:
        total += layer.keys.element_size() * layer.keys.numel()
        total += layer.values.element_size() * layer.values.numel()
    return total

# ── load ──────────────────────────────────────────────────────────────────────
print("Loading tokenizer..."); sys.stdout.flush()
tok = AutoTokenizer.from_pretrained("/home/hermes/.cache/huggingface/modules",
    trust_remote_code=True, padding_side="left")
if tok.pad_token is None: tok.pad_token = tok.eos_token_id

print("Loading model..."); sys.stdout.flush()
m = AutoModelForCausalLM.from_pretrained(
    "/home/hermes/.cache/huggingface/modules/models/Qwen3-4B-4bit",
    device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True,
)
m = m.to(device); m.eval(); torch.xpu.synchronize()
model_vram = torch.xpu.memory_allocated() / 1e9
print(f"Model ready. VRAM={model_vram:.2f}GB\n"); sys.stdout.flush()

# ── prompt ────────────────────────────────────────────────────────────────────
text = (
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
    "them to wake once more and resume their ancient work."
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
    "efficiency or longevity."
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
    "Original Libraries beneath the dried seabed of the Former Ocean."
    "As the excavation deepened, the researchers uncovered increasingly complex layers "
    "of the Empire's technological civilization. Beneath the surface ruins of the late "
    "Cataclysm period lay the remains of an even older culture — one that had apparently "
    "mastered the principles of thermal regulation and mechanical self-repair centuries "
    "before the first automata were ever constructed. The artifacts from this earlier "
    "period were smaller, more delicate, and far more numerous than anything found in "
    "the upper strata of the excavation. They suggested a society that had gradually "
    "scaled up its mechanical creations over generations, culminating in the great "
    "War Golems and the palace Songbirds that had defined the Empire's image for so long."
)
tokens = tok.encode(text, add_special_tokens=True)
if len(tokens) > 1200: tokens = tokens[:1200]
prompt = tok.decode(tokens)
inp = {k: v.to(device) for k, v in tok(prompt, return_tensors="pt").items()}
print(f"Prompt: {len(tokens)} tokens\n"); sys.stdout.flush()

# ── baseline prefill + 10-token gen ──────────────────────────────────────────
print("=== BASELINE (prefill + 10-token gen) ==="); sys.stdout.flush()
torch.xpu.reset_peak_memory_stats()
torch.xpu.synchronize(); t0 = time.time()
with torch.no_grad():
    gen_out = m.generate(
        inp["input_ids"],
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
torch.xpu.synchronize(); t_base = time.time() - t0
vram_base_peak = torch.xpu.max_memory_allocated() / 1e9
print(f"  Time:        {t_base:.1f}s")
print(f"  Peak VRAM:   {vram_base_peak:.2f}GB")
print(f"  Output:      {tok.decode(gen_out[0])[:200]}\n"); sys.stdout.flush()

# ── TurboQuant path ───────────────────────────────────────────────────────────
print("=== TURBOQUANT (prefill → compress → 10-token decode) ==="); sys.stdout.flush()

# 1. prefill
torch.xpu.reset_peak_memory_stats()
torch.xpu.synchronize(); t0 = time.time()
with torch.no_grad():
    prefill_out = m(input_ids=inp["input_ids"], use_cache=True)
torch.xpu.synchronize(); t_prefill = time.time() - t0
vram_after_prefill = torch.xpu.memory_allocated() / 1e9
pkv = prefill_out.past_key_values
n_layers, n_heads = len(pkv.layers), pkv.layers[0].keys.shape[1]
head_dim = pkv.layers[0].keys.shape[3]
seq_len = pkv.layers[0].keys.shape[2]
dtype = pkv.layers[0].keys.dtype
baseline_kv_bytes = kv_bytes(pkv)
print(f"  Prefill:        {t_prefill:.1f}s  VRAM={vram_after_prefill:.2f}GB")
print(f"  KV shape:       {n_layers}L x {n_heads}H x seq={seq_len} x dim={head_dim}")
print(f"  KV dtype:       {dtype}")
print(f"  Baseline KV:    {baseline_kv_bytes/1e6:.1f}MB  ({n_layers*2*n_heads*seq_len*head_dim*dtype.itemsize/1e6:.1f}MB theoretical)"); sys.stdout.flush()

# 2. compress
print("  Building TurboQuant config (uniform 3-bit)..."); sys.stdout.flush()
t0 = time.time()
config = TurboQuantConfig(d=head_dim, b_mse=3, device=device,
    mixed_precision=False, use_online_codebook=False)
rotations = [[config.make_rotation(l, h) for h in range(n_heads)] for l in range(n_layers)]
S_matrices = [[config.make_qjl_matrix(l, h) for h in range(n_heads)] for l in range(n_layers)]
print(f"  Precompute:      {time.time()-t0:.1f}s"); sys.stdout.flush()

print("  Compressing KV cache..."); sys.stdout.flush()
t0 = time.time()
# S_seed formula must match TurboQuantCache.__init__ exactly
def qjl_seed(l: int, h: int) -> int:
    return ((l * 1000003) ^ (h * 999979) ^ 0x5A5A5A5A) & 0xFFFFFFFF

compressed_parts = []  # store compressed data, NOT decompressed copies
debug_layer_h = (0, 0)  # (layer, head) to trace
for l in range(n_layers):
    k_parts, v_parts = [], []
    for h in range(n_heads):
        k_f = pkv.layers[l].keys[:, h:h+1, :, :].reshape(-1, head_dim).float()
        v_f = pkv.layers[l].values[:, h:h+1, :, :].reshape(-1, head_dim).float()
        S_seed = qjl_seed(l, h)
        kc = turboquant_encode_internal(k_f, config.codebook, rotations[l][h],
                                        S_matrices[l][h], S_seed)
        vc = turboquant_encode_internal(v_f, config.codebook, rotations[l][h],
                                        S_matrices[l][h], S_seed)
        k_parts.append(kc); v_parts.append(vc)
        # Debug: print size of first K and V compressed object
        if (l, h) == debug_layer_h:
            def obj_bytes(obj):
                total = 0
                for field in obj.__dataclass_fields__.values():
                    val = getattr(obj, field.name)
                    if isinstance(val, torch.Tensor):
                        total += val.element_size() * val.numel()
                    elif isinstance(val, (list, tuple)):
                        for item in val:
                            if isinstance(item, torch.Tensor):
                                total += item.element_size() * item.numel()
                return total
            kc_bytes = (kc.pq.indices.element_size() * kc.pq.indices.numel() +
                        kc.pq.norm.element_size() * kc.pq.norm.numel() +
                        kc.qjl.signs.element_size() * kc.qjl.signs.numel() +
                        kc.qjl.r_norm.element_size() * kc.qjl.r_norm.numel())
            vc_bytes = (vc.pq.indices.element_size() * vc.pq.indices.numel() +
                        vc.pq.norm.element_size() * vc.pq.norm.numel() +
                        vc.qjl.signs.element_size() * vc.qjl.signs.numel() +
                        vc.qjl.r_norm.element_size() * vc.qjl.r_norm.numel())
            k_orig = k_f.numel() * k_f.element_size()
            v_orig = v_f.numel() * v_f.element_size()
            print(f"    DEBUG Layer{l}xHead{h}: K orig={k_orig}B comp={kc_bytes}B ratio={k_orig/kc_bytes:.2f}x "
                  f"| V orig={v_orig}B comp={vc_bytes}B ratio={v_orig/vc_bytes:.2f}x")
            print(f"      kc.pq.indices: {kc.pq.indices.shape} {kc.pq.indices.dtype} "
                  f"| norm: {kc.pq.norm.shape} {kc.pq.norm.dtype}")
            print(f"      kc.qjl.signs: {kc.qjl.signs.shape} {kc.qjl.signs.dtype} "
                  f"| r_norm: {kc.qjl.r_norm.shape} {kc.qjl.r_norm.dtype}")
    compressed_parts.append((k_parts, v_parts))
    if l % 6 == 0: print(f"    Layer {l}/{n_layers}"); sys.stdout.flush()
torch.xpu.synchronize(); t_compress = time.time() - t0
print(f"  Compress time:  {t_compress:.1f}s"); sys.stdout.flush()

# 3. FREE original KV immediately — this is the critical measurement
del pkv
torch.xpu.synchronize()
vram_after_del_pkv = torch.xpu.memory_allocated() / 1e9

# 4. Measure actual compressed representation sizes (not theoretical)
# What's ACTUALLY stored in each TurboQuantCompressed object:
#   PQ:   indices [seq_len, d] uint8  = seq_len * d bytes
#         norm    [seq_len] float32  = seq_len * 4 bytes
#   QJL:  signs   [seq_len, d] uint8 = seq_len * d bytes
#         r_norm  [seq_len] float32  = seq_len * 4 bytes
#   seed + device (tiny)
# That's ~2 * seq_len * d bytes per head per layer (for K+V)
b_mse = config.b_mse
actual_compressed_bytes = 0
for l in range(n_layers):
    for h in range(n_heads):
        kc, vc = compressed_parts[l][0][h], compressed_parts[l][1][h]
        # PQ: indices + norm (one per head per layer, per K and V = 2x)
        actual_compressed_bytes += kc.pq.indices.element_size() * kc.pq.indices.numel()  # indices
        actual_compressed_bytes += kc.pq.norm.element_size() * kc.pq.norm.numel()         # norm
        actual_compressed_bytes += vc.pq.indices.element_size() * vc.pq.indices.numel()
        actual_compressed_bytes += vc.pq.norm.element_size() * vc.pq.norm.numel()
        # QJL: signs + r_norm
        actual_compressed_bytes += kc.qjl.signs.element_size() * kc.qjl.signs.numel()      # signs
        actual_compressed_bytes += kc.qjl.r_norm.element_size() * kc.qjl.r_norm.numel()    # r_norm
        actual_compressed_bytes += vc.qjl.signs.element_size() * vc.qjl.signs.numel()
        actual_compressed_bytes += vc.qjl.r_norm.element_size() * vc.qjl.r_norm.numel()
        # seed + device are scalar/int, negligible

actual_compressed_mb = actual_compressed_bytes / 1e6
print(f"\n  After del pkv + sync:         VRAM={vram_after_del_pkv:.2f}GB")
print(f"  ACTUAL compressed size (PQ+QJL tensors): {actual_compressed_mb:.1f}MB")
print(f"  Theoretical FP16 KV:         {baseline_kv_bytes/1e6:.1f}MB")
print(f"  Actual compression ratio:    {baseline_kv_bytes/actual_compressed_bytes:.1f}x")
print(f"  Theoretical (16/{b_mse}):   {16/b_mse:.1f}x"); sys.stdout.flush()

# 5. Rebuild cache from decompressed (needed for decode)
print("\n  Rebuilding DynamicCache from decompressed tensors..."); sys.stdout.flush()
rebuilt_kv = []
for l in range(n_layers):
    new_k_parts, new_v_parts = [], []
    for h in range(n_heads):
        kc, vc = compressed_parts[l][0][h], compressed_parts[l][1][h]
        kr = polarquant_decode(kc.pq)[..., :head_dim].reshape(1, 1, seq_len, head_dim).to(dtype).contiguous()
        vr = polarquant_decode(vc.pq)[..., :head_dim].reshape(1, 1, seq_len, head_dim).to(dtype).contiguous()
        new_k_parts.append(kr); new_v_parts.append(vr)
    rebuilt_kv.append((torch.cat(new_k_parts, dim=1), torch.cat(new_v_parts, dim=1)))
torch.xpu.synchronize()
vram_rebuilt = torch.xpu.memory_allocated() / 1e9
print(f"  After rebuild:   VRAM={vram_rebuilt:.2f}GB"); sys.stdout.flush()

cache = DynamicCache()
for l in range(n_layers):
    cache.update(rebuilt_kv[l][0], rebuilt_kv[l][1], layer_idx=l)

# free decompressed copies
del rebuilt_kv
torch.xpu.synchronize()
vram_after_free_decomp = torch.xpu.memory_allocated() / 1e9
print(f"  After free rebuilt: VRAM={vram_after_free_decomp:.2f}GB\n"); sys.stdout.flush()

# 6. decode loop
logits = prefill_out.logits[:, -1, :]
next_tok = logits.argmax(dim=-1, keepdim=True)
gen_ids = torch.cat([inp["input_ids"], next_tok], dim=-1)

print("  Decoding 10 tokens..."); sys.stdout.flush()
torch.xpu.synchronize(); t0 = time.time()
for i in range(10):
    with torch.no_grad():
        o = m(input_ids=next_tok, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        next_tok = o.logits[:, -1:, :].argmax(dim=-1)
        gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
torch.xpu.synchronize(); t_decode = time.time() - t0
vram_tq_peak = torch.xpu.max_memory_allocated() / 1e9
print(f"  Decode:         {t_decode:.1f}s  ({(t_decode/10)*1000:.0f}ms/token)"); sys.stdout.flush()

# ── results ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
ratio = baseline_kv_bytes / actual_compressed_bytes if actual_compressed_bytes > 0 else 0
print(f"\n  Model:        Qwen3-4B-4bit on Intel Arc B580")
print(f"  Context:      {len(tokens)} tokens")
print(f"  Generation:   10 new tokens")
print(f"  Compression:  {b_mse}-bit TurboQuant (theoretical ratio: {16/b_mse:.1f}x)")
print(f"\n  {'Metric':<35} {'Value':<15}")
print(f"  {'-'*35} {'-'*15}")
print(f"  {'Baseline peak VRAM':<35} {vram_base_peak:<15.2f}GB")
print(f"  {'Baseline KV cache bytes':<35} {baseline_kv_bytes/1e6:<15.1f}MB")
print(f"  {'ACTUAL compressed bytes (PQ+QJL)':<35} {actual_compressed_mb:<15.1f}MB")
print(f"  {'VRAM after del pkv + sync':<35} {vram_after_del_pkv:<15.2f}GB")
print(f"  {'VRAM after cache rebuild':<35} {vram_rebuilt:<15.2f}GB")
print(f"  {'VRAM after freeing rebuilt':<35} {vram_after_free_decomp:<15.2f}GB")
print(f"  {'TurboQuant peak during decode':<35} {vram_tq_peak:<15.2f}GB")
print(f"\n  {'Baseline time':<35} {t_base:<15.1f}s")
print(f"  {'TurboQuant total time':<35} {t_prefill+t_compress+t_decode:<15.1f}s")
print(f"  {'  - Prefill':<33} {t_prefill:.1f}s")
print(f"  {'  - Compress':<33} {t_compress:.1f}s")
print(f"  {'  - Decode 10 tokens':<30} {t_decode:.1f}s")
print(f"\n  KV compression ratio (theoretical): {ratio:.1f}x")
print(f"  VRAM saved vs baseline peak:        {vram_base_peak - vram_tq_peak:.2f}GB  "
      f"({'BETTER' if vram_tq_peak < vram_base_peak else 'WORSE' if vram_tq_peak > vram_base_peak else 'SAME'})")
print(f"\n  TurboQuant output:\n  {tok.decode(gen_ids[0])}")
print("\n" + "=" * 70)
