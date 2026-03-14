import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import softmax, layer_norm
import os
import sys

def mha_scores(x, w_qkv, b_qkv, n_head):
    """
    Multi-Head Attention の重み (Attention Scores) を計算する。
    (出力ベクトルではなく、どの単語にどれだけ注目しているかの分布を返す)
    """
    batch_size, seq_len, embed_dim = x.shape
    qkv = np.matmul(x, w_qkv) + b_qkv
    q, k, v = np.split(qkv, 3, axis=-1)

    # Multi-head への分割
    d_k = embed_dim // n_head
    q = q.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, seq_len, n_head, d_k).transpose(0, 2, 1, 3)

    # スコア計算: (q @ k.T) / sqrt(d_k)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    # 因果マスキング (Causal Masking)
    mask = np.tril(np.ones((seq_len, seq_len)))
    scores = np.where(mask == 0, -1e10, scores)

    # 正規化 (Softmax)
    probs = softmax(scores)
    return probs

def main():
    model_id = "openai-community/gpt2"
    if not os.path.exists(f"weights/{model_id}"):
        print(f"Error: weights/{model_id} が見つかりません。")
        sys.exit(1)

    print("--- Attention 機構体験 (GPT-2 第1層) ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)
    tokenizer = Tokenizer(model_id)
    
    # 体験用の短い文
    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([i]) for i in input_ids]
    
    # 1. Embedding 層
    # wte + wpe (第0層からの入力)
    x = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
    x = x[np.newaxis, ...] # Batch 次元追加 (1, seq_len, 768)

    # 2. 第1層の LayerNorm を適用 (GPT-2 は Pre-LayerNorm 方式)
    ln_1 = params.blocks[0].ln_1
    x_norm = layer_norm(x, ln_1)

    # 3. Attention 重みの計算
    attn_params = params.blocks[0].attn
    n_head = 12
    # probs: (batch, head, seq_len, seq_len)
    probs = mha_scores(x_norm, attn_params.w_qkv, attn_params.b_qkv, n_head)

    print(f"\n入力文: '{text}'")
    print(f"トークン数: {len(tokens)}")
    
    # ヘッド 0 (最初の視点) の重みを詳しく見る
    head_idx = 0
    p = probs[0, head_idx]
    
    print(f"\n[Head {head_idx}] の注目度マトリックス (一部):")
    # 最後の単語 ' dog.' がどの単語に注目しているかを表示
    target_idx = len(tokens) - 1
    print(f"最後のトークン '{tokens[target_idx]}' が注目している先:")
    
    # 注目度が高い順にソートして表示
    sorted_indices = np.argsort(p[target_idx])[::-1]
    for i in sorted_indices:
        score = p[target_idx][i]
        if score > 0.01: # 1% 以上の注目度があるものだけ
            print(f"  -> {tokens[i]:10} : {score:.4f}")

    print("\n[Causal Masking の確認]")
    # 最初の単語 'The' が未来を見ているか確認
    print(f"最初のトークン '{tokens[0]}' が注目している先:")
    for i, score in enumerate(p[0]):
        if score > 0:
            print(f"  -> {tokens[i]:10} : {score:.4f}")

    print("\n--- 体験終了 ---")
    print("Attention によって、各トークンのベクトルは『注目先』の情報を取り込んでアップデートされます。")

if __name__ == "__main__":
    main()
