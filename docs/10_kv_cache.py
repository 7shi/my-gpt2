import numpy as np
import time
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import GPT2
import os
import sys

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- KV キャッシュ ---")
print("重みをロード中...")
params = load_gpt2_weights(model_id)
tokenizer = Tokenizer(model_id)
model = GPT2(params, n_head=12)

prompt = "Artificial Intelligence will"
input_ids = tokenizer.encode(prompt)

# 1. KV キャッシュなしの生成（従来方式）
print("\n" + "=" * 50)
print("1. KV キャッシュなし（毎回全トークンを再計算）")
n_steps = 10
ids_no_cache = input_ids.copy()

start = time.time()
for step in range(n_steps):
    logits = model(np.array(ids_no_cache))
    next_token = int(np.argmax(logits[-1, :]))
    ids_no_cache.append(next_token)
elapsed_no_cache = time.time() - start

text_no_cache = tokenizer.decode(ids_no_cache)
print(f"  生成結果: '{text_no_cache}'")
print(f"  所要時間: {elapsed_no_cache:.3f} 秒")

# 2. KV キャッシュありの生成
print("\n" + "=" * 50)
print("2. KV キャッシュあり（新トークンのみ計算）")
ids_cached = input_ids.copy()

start = time.time()
# Prefill
inputs = np.array(input_ids)
logits, kv_cache = model(inputs, kv_cache=None)
next_token = int(np.argmax(logits[-1, :]))
ids_cached.append(next_token)

# Incremental
for step in range(n_steps - 1):
    inputs = np.array([ids_cached[-1]])
    logits, kv_cache = model(inputs, kv_cache=kv_cache)
    next_token = int(np.argmax(logits[-1, :]))
    ids_cached.append(next_token)
elapsed_cached = time.time() - start

text_cached = tokenizer.decode(ids_cached)
print(f"  生成結果: '{text_cached}'")
print(f"  所要時間: {elapsed_cached:.3f} 秒")

# 3. 結果の一致を確認
print("\n" + "=" * 50)
print("3. 結果の一致確認")
match = ids_no_cache == ids_cached
print(f"  トークン列一致: {match}")
print(f"  高速化: {elapsed_no_cache / elapsed_cached:.1f}x")

# 4. KV キャッシュの中身
print("\n" + "=" * 50)
print("4. KV キャッシュの構造")
print(f"  層数: {len(kv_cache)}")
k, v = kv_cache[0]
print(f"  Layer 0 の K の形状: {k.shape}  (batch, n_head, seq_len, head_size)")
print(f"  Layer 0 の V の形状: {v.shape}")
total_bytes = sum(k.nbytes + v.nbytes for k, v in kv_cache)
print(f"  全層のキャッシュサイズ: {total_bytes:,} bytes ({total_bytes / 1024:.1f} KB)")

# 5. ステップごとの計算量の違い
print("\n" + "=" * 50)
print("5. ステップごとの計算量")
print()
print("  キャッシュなし（毎回全トークン）:")
prompt_len = len(input_ids)
for step in range(5):
    seq_len = prompt_len + step
    print(f"    Step {step}: seq_len={seq_len}, Attention 計算量 ∝ {seq_len}×{seq_len} = {seq_len*seq_len}")
print()
print("  キャッシュあり（新トークンのみ）:")
print(f"    Prefill: seq_len={prompt_len}, Attention 計算量 ∝ {prompt_len}×{prompt_len} = {prompt_len*prompt_len}")
for step in range(1, 5):
    kv_len = prompt_len + step
    print(f"    Step {step}: q_len=1, kv_len={kv_len}, Attention 計算量 ∝ 1×{kv_len} = {kv_len}")
