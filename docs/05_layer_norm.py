import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import LayerNormParams
import os
import sys

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- Layer Normalization ---")
print("重みをロード中...")
params = load_gpt2_weights(model_id)
tokenizer = Tokenizer(model_id)

# 入力テキスト
text = "The capital of France is"
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

# Embedding
x = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
x = x[np.newaxis, ...]  # (1, seq_len, 768)

print(f"\n入力: '{text}'")
print(f"トークン数: {len(tokens)}")

# 1. LayerNorm 適用前の統計
print("\n" + "=" * 50)
print("1. LayerNorm 適用前（Embedding 直後）")
print(f"  {'トークン':>12}  {'平均':>10}  {'分散':>10}")
for i, tok in enumerate(tokens):
    vec = x[0, i]
    print(f"  {tok:>12}  {np.mean(vec):10.4f}  {np.var(vec):10.4f}")

# 2. 正規化のみ（γ/β なし）
print("\n" + "=" * 50)
print("2. 正規化のみ（平均0・分散1に変換、γ/β 適用前）")
eps = 1e-5
mean = np.mean(x, axis=-1, keepdims=True)
variance = np.var(x, axis=-1, keepdims=True)
x_normalized = (x - mean) / np.sqrt(variance + eps)
print(f"  {'トークン':>12}  {'平均':>10}  {'分散':>10}")
for i, tok in enumerate(tokens):
    vec = x_normalized[0, i]
    print(f"  {tok:>12}  {np.mean(vec):10.6f}  {np.var(vec):10.6f}")

# 3. LayerNorm（γ/β 適用後）
print("\n" + "=" * 50)
print("3. LayerNorm 適用後（γ でスケール、β でシフト）")
ln_1 = params.blocks[0].ln_1
x_ln = ln_1(x)
print(f"  {'トークン':>12}  {'平均':>10}  {'分散':>10}")
for i, tok in enumerate(tokens):
    vec = x_ln[0, i]
    print(f"  {tok:>12}  {np.mean(vec):10.4f}  {np.var(vec):10.4f}")

# 4. γ と β の統計
print("\n" + "=" * 50)
print("4. 学習済み γ（gain）と β（bias）の統計")
print(f"  γ: 平均={np.mean(ln_1.g):.4f}, 標準偏差={np.std(ln_1.g):.4f}, "
      f"最小={np.min(ln_1.g):.4f}, 最大={np.max(ln_1.g):.4f}")
print(f"  β: 平均={np.mean(ln_1.b):.4f}, 標準偏差={np.std(ln_1.b):.4f}, "
      f"最小={np.min(ln_1.b):.4f}, 最大={np.max(ln_1.b):.4f}")

# 5. GPT-2 での使用箇所
print("\n" + "=" * 50)
print("5. GPT-2 における LayerNorm の使用箇所")
print(f"  各ブロック: ln_1（Attention 前）, ln_2（MLP 前）× 12ブロック = 24回")
print(f"  最終出力:   ln_f（LM Head 前）× 1回")
print(f"  合計: 25回の LayerNorm")
