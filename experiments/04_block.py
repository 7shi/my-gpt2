import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import layer_norm, mha, mlp
import os
import sys

def main():
    model_id = "openai-community/gpt2"
    if not os.path.exists(f"weights/{model_id}"):
        print(f"Error: weights/{model_id} が見つかりません。")
        sys.exit(1)

    print("--- Transformer Block 体験 (GPT-2 第1層) ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)
    tokenizer = Tokenizer(model_id)
    
    # 体験用の短い文
    text = "Machine Learning"
    input_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([i]) for i in input_ids]
    
    # 0. Embedding 層 (入力準備)
    # wte + wpe (第0層からの入力)
    x_input = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
    x_input = x_input[np.newaxis, ...] # (1, seq_len, 768)
    
    block_params = params.blocks[0]
    n_head = 12

    print(f"\n入力文: '{text}' (トークン数: {len(tokens)})")
    print("-" * 40)

    # 1. Attention ステップ (Pre-LayerNorm + MHA + Residual)
    # LayerNorm
    x_norm_1 = layer_norm(x_input, block_params.ln_1)
    # MHA
    x_attn = mha(x_norm_1, block_params.attn, n_head=n_head)
    # Residual Connection
    x_after_attn = x_input + x_attn

    print("Step 1: Attention (文脈の統合)")
    print(f"  入力ベクトルの標準偏差: {np.std(x_input):.4f}")
    print(f"  Attention 出力の標準偏差: {np.std(x_attn):.4f}")
    print(f"  残差接続後の標準偏差: {np.std(x_after_attn):.4f}")

    # 2. MLP ステップ (Pre-LayerNorm + MLP + Residual)
    # LayerNorm
    x_norm_2 = layer_norm(x_after_attn, block_params.ln_2)
    # MLP
    x_mlp = mlp(x_norm_2, block_params.mlp)
    # Residual Connection
    x_after_mlp = x_after_attn + x_mlp

    print("\nStep 2: MLP (特徴の抽出と深掘り)")
    print(f"  MLP 出力の標準偏差: {np.std(x_mlp):.4f}")
    print(f"  最終出力の標準偏差: {np.std(x_after_mlp):.4f}")

    print("-" * 40)
    print("\n[考察]")
    print("1. LayerNorm によって、計算が不安定にならないよう数値が正規化されます。")
    print("2. 残差接続 (x + output) によって、元の情報を保持したまま『差分』だけが追加されます。")
    print("3. MLP は各トークンを独立して処理し、高次元空間（3072次元）に投影して複雑な特徴を抽出します。")

    print("\n--- 体験終了 ---")
    print("GPT-2 はこの Block を 12 回繰り返すことで、高度な文章理解を実現しています。")

if __name__ == "__main__":
    main()
