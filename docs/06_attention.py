import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import softmax
import os
import sys

def mha_scores(x, w_qkv, b_qkv, n_head):
    """Multi-Head Attention のスコア（注目度分布）を計算する。"""
    batch_size, seq_len, embed_dim = x.shape
    qkv = x @ w_qkv + b_qkv
    q, k, v = np.split(qkv, 3, axis=-1)

    d_k = embed_dim // n_head
    q = q.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)

    # 因果マスキング
    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 0, -1e10, scores)

    return softmax(scores)

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- Attention 機構 ---")
print("重みをロード中...")
params = load_gpt2_weights(model_id)
tokenizer = Tokenizer(model_id)
n_head = 12

# 入力テキスト
text = "The quick brown fox jumps over the lazy dog."
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

# Embedding + LayerNorm
x = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
x = x[np.newaxis, ...]
x_norm = params.blocks[0].ln_1(x)

# Attention スコアを計算
attn_params = params.blocks[0].attn
probs = mha_scores(x_norm, attn_params.w_qkv, attn_params.b_qkv, n_head)

print(f"\n入力: '{text}'")
print(f"トークン: {tokens}")
print(f"トークン数: {len(tokens)}")

# 1. 因果マスキングの確認
print("\n" + "=" * 50)
print("1. 因果マスキングの確認（Head 0）")
p = probs[0, 0]
print(f"  最初のトークン '{tokens[0]}' の注目先:")
for i, score in enumerate(p[0]):
    if score > 0.001:
        print(f"    -> {tokens[i]:10} : {score:.4f}")
print("  （自分より後のトークンは見えないため、自分だけに100%注目）")

# 2. 最後のトークンの注目先
print("\n" + "=" * 50)
print(f"2. 最後のトークン '{tokens[-1]}' の注目先（Head 0）")
target_idx = len(tokens) - 1
sorted_indices = np.argsort(p[target_idx])[::-1]
for i in sorted_indices:
    score = p[target_idx][i]
    if score > 0.01:
        print(f"    -> {tokens[i]:10} : {score:.4f}")

# 3. ヘッド間の注目パターンの違い
print("\n" + "=" * 50)
print(f"3. 複数ヘッドの注目パターン比較（最後のトークン '{tokens[-1]}' の注目先 上位3）")
for head in range(min(4, n_head)):
    p_head = probs[0, head, target_idx]
    top3 = np.argsort(p_head)[::-1][:3]
    items = ", ".join(f"{tokens[i]}({p_head[i]:.3f})" for i in top3)
    print(f"  Head {head}: {items}")
