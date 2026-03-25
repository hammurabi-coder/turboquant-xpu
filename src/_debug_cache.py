from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
inputs = tokenizer("Hello world", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**inputs, use_cache=True)
kv = out.past_key_values
print(f"Type: {type(kv)}")
print(f"Len: {len(kv)}")
for i, item in enumerate(kv):
    t = type(item).__name__
    if isinstance(item, (tuple, list)):
        print(f"Layer {i}: {t} of len {len(item)}")
        for j, sub in enumerate(item):
            if hasattr(sub, "shape"):
                print(f"  [{j}] {sub.shape} {sub.dtype}")
    elif hasattr(item, "shape"):
        print(f"Layer {i}: tensor {item.shape}")
    else:
        print(f"Layer {i}: {t}")
    if i >= 2:
        print("...")
        break
# Also try accessing via .layers
if hasattr(kv, "layers"):
    print(f"\nkv.layers type: {type(kv.layers)}, len: {len(kv.layers)}")
    if kv.layers:
        layer0 = kv.layers[0]
        print(f"layers[0] type: {type(layer0).__name__}")
        if hasattr(layer0, "key_states"):
            print(f"  key_states: {layer0.key_states.shape}")
        if hasattr(layer0, "value_states"):
            print(f"  value_states: {layer0.value_states.shape}")
