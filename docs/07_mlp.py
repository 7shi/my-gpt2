import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import gelu
import os
import sys

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- MLP（Multi-Layer Perceptron）体験 ---")
print("重みをロード中...")
params = load_gpt2_weights(model_id)
tokenizer = Tokenizer(model_id)
n_head = 12

# 入力テキスト
text = "The capital of France is"
input_ids = tokenizer.encode(text)
tokens = [tokenizer.decode([i]) for i in input_ids]

# Embedding → LayerNorm → Attention → 残差接続 → LayerNorm（MLP の入力を準備）
x = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
x = x[np.newaxis, ...]
block = params.blocks[0]
x_after_attn = x + block.attn(block.ln_1(x), n_head=n_head)
x_ln2 = block.ln_2(x_after_attn)

print(f"\n入力: '{text}'")
print(f"トークン: {tokens}")

# 1. MLP の次元変化
print("\n" + "=" * 50)
print("1. MLP の次元変化: 768 → 3072 → 768")
# 中間層を手動で計算
intermediate = np.matmul(x_ln2, block.mlp.w_fc) + block.mlp.b_fc
print(f"  入力:   {x_ln2.shape}  （768次元）")
print(f"  中間層: {intermediate.shape}  （3072次元 = 768 × 4）")
activated = gelu(intermediate)
output = np.matmul(activated, block.mlp.w_proj) + block.mlp.b_proj
print(f"  出力:   {output.shape}  （768次元に戻る）")

# 2. GELU 活性化の効果
print("\n" + "=" * 50)
print("2. GELU 活性化の効果（中間層 3072 次元）")
total = intermediate.size
near_zero = np.sum(np.abs(activated) < 0.1)
negative_input = np.sum(intermediate < 0)
suppressed = np.sum((intermediate < 0) & (np.abs(activated) < 0.01))
print(f"  中間層の要素数: {total}")
print(f"  負の入力: {negative_input} ({100*negative_input/total:.1f}%)")
print(f"  GELU で抑制（|出力| < 0.01）: {suppressed} ({100*suppressed/total:.1f}%)")
print(f"  活性化後にほぼゼロ（|出力| < 0.1）: {near_zero} ({100*near_zero/total:.1f}%)")

# 3. MLP 前後のベクトル変化
print("\n" + "=" * 50)
print("3. MLP 前後のベクトル変化")
x_mlp = block.mlp(x_ln2)
print(f"  {'トークン':>12}  {'入力std':>8}  {'出力std':>8}  {'コサイン類似度':>14}")
for i, tok in enumerate(tokens):
    v_in = x_ln2[0, i]
    v_out = x_mlp[0, i]
    sim = cosine_similarity(v_in, v_out)
    print(f"  {tok:>12}  {np.std(v_in):8.4f}  {np.std(v_out):8.4f}  {sim:14.4f}")

print("\n--- 体験終了 ---")
