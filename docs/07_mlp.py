import numpy as np
import unicodedata
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import gelu
import os
import sys

def display_width(s):
    """東アジア文字（全角）を2カラム、それ以外を1カラムとして文字列の表示幅を返す。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

def center_w(s, w):
    pad = max(0, w - display_width(s))
    left = pad // 2
    return ' ' * left + s + ' ' * (pad - left)

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- MLP（Multi-Layer Perceptron） ---")
print("重みをロード中...")
tokenizer = Tokenizer(model_id)
model = load_gpt2_weights(model_id)

# 入力テキスト
text = "The capital of France is"
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

# Embedding → LayerNorm → Attention → 残差接続 → LayerNorm（MLP の入力を準備）
x = model.wte[input_ids] + model.wpe[np.arange(len(input_ids))]
block = model.blocks[0]
x_after_attn = x + block.attn(block.ln_1(x))
x_ln2 = block.ln_2(x_after_attn)

print(f"\n入力: '{text}'")
print(f"トークン: {tokens}")

# GELU の入出力例
print("\n" + "=" * 50)
print("GELU の入出力例")
cols_gelu = [("入力", 6), ("GELU 出力", 10)]
print("| " + " | ".join(center_w(h, w) for h, w in cols_gelu) + " |")
print("|" + "|".join("-" * (w + 2) for _, w in cols_gelu) + "|")
for val in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
    print(f"| {val:>6.1f} | {gelu(np.array(val)):>10.4f} |")

# 1. MLP の次元変化
print("\n" + "=" * 50)
print("1. MLP の次元変化: 768 → 3072 → 768")
# 中間層を手動で計算
intermediate = x_ln2 @ block.mlp.w_fc + block.mlp.b_fc
print(f"  入力:   {x_ln2.shape}  （768次元）")
print(f"  中間層: {intermediate.shape}  （3072次元 = 768 × 4）")
activated = gelu(intermediate)
output = activated @ block.mlp.w_proj + block.mlp.b_proj
print(f"  出力:   {output.shape}  （768次元に戻る）")

n_dim = intermediate.shape[1]
cols1 = [("トークン", 12), ("正の成分", 9), ("負の成分", 9)]
print("| " + " | ".join(center_w(h, w) for h, w in cols1) + " |")
print("|" + "|".join("-" * (w + 2) for _, w in cols1) + "|")
for i, token in enumerate(tokens):
    pos = np.sum(intermediate[i] > 0)
    neg = np.sum(intermediate[i] < 0)
    token_s = f"`{token!r}`"
    print(f"| {token_s:12} | {pos:>4d}/{n_dim} | {neg:>4d}/{n_dim} |")

# 2. MLP 前後のベクトル変化
print("\n" + "=" * 50)
print("2. MLP 前後のベクトル変化")
x_mlp = block.mlp(x_ln2)
cols3 = [("トークン", 12), ("入力std", 8), ("出力std", 8), ("コサイン類似度", 14)]
print("| " + " | ".join(center_w(h, w) for h, w in cols3) + " |")
print("|" + "|".join("-" * (w + 2) for _, w in cols3) + "|")
for i, tok in enumerate(tokens):
    v_in = x_ln2[i]
    v_out = x_mlp[i]
    sim = cosine_similarity(v_in, v_out)
    tok_s = f"`{tok!r}`"
    print(f"| {tok_s:12} | {np.std(v_in):8.4f} | {np.std(v_out):8.4f} | {sim:14.4f} |")
