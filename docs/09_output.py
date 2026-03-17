import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import softmax
import os
import sys

def get_top_tokens(logits, tokenizer, top_k=5):
    """ロジットから上位 k 個のトークンとその確率を返す。"""
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(tokenizer.decode([int(idx)]), probs[idx]) for idx in top_indices]

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- 出力とサンプリング ---")
print("重みをロード中...")
tokenizer = Tokenizer(model_id)
model = load_gpt2_weights(model_id)

prompt = "Artificial Intelligence will"
input_ids = tokenizer.encode(prompt)

print(f"\nプロンプト: '{prompt}'")

# 1. Temperature による確率分布の変化
print("\n" + "=" * 50)
print("1. Temperature による確率分布の変化")
logits = model(np.array(input_ids))
next_logits = logits[-1, :]

for temp in [0.5, 1.0, 2.0]:
    scaled = next_logits / temp
    probs = softmax(scaled)
    top_indices = np.argsort(probs)[::-1][:5]
    items = ", ".join(f"'{tokenizer.decode([int(i)])}' {probs[i]:.3f}" for i in top_indices)
    label = {0.5: "集中", 1.0: "通常", 2.0: "平坦"}[temp]
    print(f"  T={temp}（{label}）: {items}")

# 2. 自己回帰の様子
print("\n" + "=" * 50)
print("2. 自己回帰生成（Greedy、3ステップ）")
current_ids = input_ids.copy()

for step in range(3):
    logits = model(np.array(current_ids))
    next_logits = logits[-1, :]

    top = get_top_tokens(next_logits, tokenizer)
    current_text = tokenizer.decode(current_ids)
    print(f"\n  Step {step + 1}: '{current_text}'")
    top_str = ", ".join(f"'{t}' ({p:.3f})" for t, p in top[:3])
    print(f"    → 候補: {top_str}...")

    next_token = int(np.argmax(next_logits))
    current_ids.append(next_token)
    print(f"    → 選択: '{tokenizer.decode([next_token])}'")

final = tokenizer.decode(current_ids)
print(f"\n  生成結果: '{final}'")

# 3. Weight Tying の確認
print("\n" + "=" * 50)
print("3. Weight Tying: 入力と出力で同じ行列を使用")
print(f"  WTE 形状: {model.wte.shape}  (入力: トークンID → ベクトル)")
print(f"  LM Head:  WTE.T = {model.wte.T.shape}  (出力: ベクトル → ロジット)")
print(f"  共有パラメータ数: {model.wte.size:,} ({model.wte.size * 4 / 1024 / 1024:.1f} MB)")
