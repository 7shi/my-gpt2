import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
import os
import sys

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- Layer Normalization ---")
print("重みをロード中...")
tokenizer = Tokenizer(model_id)
model = load_gpt2_weights(model_id)

# 入力テキスト
text = "The capital of France is"
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

# Embedding
x = model.wte[input_ids] + model.wpe[np.arange(len(input_ids))]
x = x[np.newaxis, ...]  # (1, seq_len, 768)

print(f"\n入力: '{text}'")
print(f"トークン数: {len(tokens)}")

# 1. LayerNorm 適用前後の統計（統合テーブル）
print("\n" + "=" * 50)
print("1. LayerNorm 適用前後（Embedding 直後 vs γ/β 適用後）")
ln_1 = model.blocks[0].ln_1
x_ln = ln_1(x)
print(f"|   トークン   | 平均(前) | 分散(前) | 平均(後) | 分散(後) |")
print(f"|--------------|---------:|---------:|---------:|---------:|")
for i, tok in enumerate(tokens):
    vec_before = x[0, i]
    vec_after = x_ln[0, i]
    tok_s = f"`{tok!r}`"
    print(f"| {tok_s:<12} | {np.mean(vec_before):8.4f} | {np.var(vec_before):8.4f} | {np.mean(vec_after):8.4f} | {np.var(vec_after):8.4f} |")

# 2. 正規化のみ（γ/β なし）
print("\n" + "=" * 50)
print("2. 正規化のみ（平均0・分散1に変換、γ/β 適用前）")
eps = 1e-5
mean = np.mean(x, axis=-1, keepdims=True)
variance = np.var(x, axis=-1, keepdims=True)
x_normalized = (x - mean) / np.sqrt(variance + eps)
print(f"|   トークン   | {'平均':>10} | {'分散':>10} |")
print(f"|--------------|-------------:|-------------:|")
for i, tok in enumerate(tokens):
    vec = x_normalized[0, i]
    tok_s = f"`{tok!r}`"
    print(f"| {tok_s:<12} | {np.mean(vec):12.6f} | {np.var(vec):12.6f} |")

# 3. γ と β の統計
print("\n" + "=" * 50)
print("3. 学習済み γ（gain）と β（bias）の統計")
print(f"  γ: 平均={ np.mean(ln_1.g): .4f}, 分散={np.var(ln_1.g):.4f}, "
      f"最小={ np.min(ln_1.g): .4f}, 最大={ np.max(ln_1.g): .4f}")
print(f"  β: 平均={ np.mean(ln_1.b): .4f}, 分散={np.var(ln_1.b):.4f}, "
      f"最小={ np.min(ln_1.b): .4f}, 最大={ np.max(ln_1.b): .4f}")
