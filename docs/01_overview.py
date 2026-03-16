import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import GPT2, TransformerBlock, softmax
import os
import sys

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- GPT-2 推論パイプライン全体像 ---")
print("重みをロード中...")
params = load_gpt2_weights(model_id)
tokenizer = Tokenizer(model_id)
n_head = 12

# 入力テキスト
text = "The capital of France is"
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

print(f"\n入力: '{text}'")
print(f"トークン: {tokens}")
print(f"トークンID: {input_ids}")

# ステップ 1: Embedding
print("\n" + "=" * 50)
print("Step 1: Embedding (トークン埋め込み + 位置埋め込み)")
x = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
print(f"  形状: {x.shape}  (トークン数, 埋め込み次元)")
print(f"  平均: {np.mean(x):.4f}, 標準偏差: {np.std(x):.4f}")

# ステップ 2: Transformer Block × 12
print("\n" + "=" * 50)
print("Step 2: Transformer Block × 12")
print(f"| {'層'} | {'平均'} | {'標準偏差'} |")
print(f"|---|---|---|")
for i, block_params in enumerate(params.blocks):
    block = TransformerBlock(block_params, n_head)
    x = block(x)
    print(f"| {i} | {np.mean(x):.4f} | {np.std(x):.4f} |")

# ステップ 3: 最終 LayerNorm
print("\n" + "=" * 50)
print("Step 3: 最終 LayerNorm")
x = params.ln_f(x)
print(f"  形状: {x.shape}")
print(f"  平均: {np.mean(x):.4f}, 標準偏差: {np.std(x):.4f}")

# ステップ 4: LM Head (Weight Tying)
print("\n" + "=" * 50)
print("Step 4: LM Head (Weight Tying: x @ wte.T)")
logits = np.matmul(x, params.wte.T)
print(f"  形状: {logits.shape}  (トークン数, 語彙数)")

# 最後のトークンの予測結果
next_token_logits = logits[-1, :]
probs = softmax(next_token_logits)
top_indices = np.argsort(probs)[::-1][:5]

print("\n" + "=" * 50)
print(f"予測結果: '{text}' の次のトークン上位5件")
for i, idx in enumerate(top_indices):
    token = tokenizer.decode([int(idx)])
    print(f"  {i+1}. '{token}' (確率: {probs[idx]:.4f})")
