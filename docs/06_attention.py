import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import softmax
import os
import sys

def mha_attention(x, w_qkv, b_qkv, n_head):
    """Multi-Head Attention のスコアと出力を計算する。"""
    batch_size, seq_len, embed_dim = x.shape
    qkv = x @ w_qkv + b_qkv
    q, k, v = np.split(qkv, 3, axis=-1)

    d_k = embed_dim // n_head
    q = q.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)

    scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)

    # 因果マスキング
    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 0, -1e10, scores)

    probs = softmax(scores)
    out = probs @ v
    return probs, v, out

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- Attention 機構 ---")
print("重みをロード中...")
tokenizer = Tokenizer(model_id)
model = load_gpt2_weights(model_id)
n_head = model.config["n_head"]

# 入力テキスト
text = "The capital of France is"
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

# Embedding + LayerNorm
x = model.wte[input_ids] + model.wpe[np.arange(len(input_ids))]
x = x[np.newaxis, ...]
x_norm = model.blocks[0].ln_1(x)

# Attention スコアを計算
attn = model.blocks[0].attn
probs, v, out = mha_attention(x_norm, attn.w_qkv, attn.b_qkv, n_head)

print(f"\n入力: '{text}'")
print(f"トークン: {tokens}")
print(f"トークン数: {len(tokens)}")

# 1. 注目度行列（Head 0）をテーブル出力
print("\n" + "=" * 50)
print("1. 注目度行列（Head 0）")
p = probs[0, 0]
tok_labels = [repr(t) for t in tokens]
col_width = max(len(l) for l in tok_labels)
header = "|" + " " * col_width + "|" + "|".join(f"{l:>{col_width}}" for l in tok_labels) + "|"
sep = "|" + "-" * col_width + "|" + "|".join("-" * (col_width - 1) + ":" for _ in tok_labels) + "|"
print(header)
print(sep)
for i, row_label in enumerate(tok_labels):
    cells = []
    for j in range(len(tokens)):
        if j > i:
            cells.append(" " * col_width)
        else:
            cells.append(f"{p[i][j]:.4f}".rjust(col_width))

    print(f"|{row_label:>{col_width}}|" + "|".join(cells) + "|")

target_idx = len(tokens) - 1

# 2. Value の重み付き平均（Head 0）
print("\n" + "=" * 50)
print(f"2. Value の重み付き平均（Head 0, 最後のトークン {tokens[-1]!r}）")
v0 = v[0, 0]  # (seq_len, d_k)
out0 = out[0, 0]  # (seq_len, d_k)
print(f"  各トークンの V（先頭5次元）:")
for i, t in enumerate(tokens):
    print(f"    {t!r:>12}: [{', '.join(f'{x:+.4f}' for x in v0[i][:5])}, ...]")
print(f"  加重平均の結果（先頭5次元）:")
print(f"    {tokens[-1]!r:>12}: [{', '.join(f'{x:+.4f}' for x in out0[target_idx][:5])}, ...]")
# 先頭次元の加重平均を展開して表示
print(f"  先頭次元の計算過程:")
total = 0.0
for j, t in enumerate(tokens):
    prob = p[target_idx][j]
    val = v0[j][0]
    prod = prob * val
    total += prod
    print(f"     {val:+.4f} * {prob:.4f} = {prod:+.7f}  ({t!r})")
print(f"    {'':>17}計 {total:+.4f}")

# 3. ヘッド間の注目パターンの違い
print("\n" + "=" * 50)
print(f"3. 複数ヘッドの注目パターン比較（最後のトークン {tokens[-1]!r} の注目先 上位3）")
for head in range(min(4, n_head)):
    p_head = probs[0, head, target_idx]
    top3 = np.argsort(p_head)[::-1][:3]
    items = ", ".join(f"{tokens[i]!r}({p_head[i]:.3f})" for i in top3)
    print(f"  Head {head}: {items}")
