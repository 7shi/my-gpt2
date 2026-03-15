import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
import os
import sys

def cosine_similarity(v1, v2):
    """2つのベクトルのコサイン類似度を計算する。"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_nearest(target_vec, wte, top_k=5):
    """対象ベクトルに最も近いトークンを探す。"""
    dot_products = np.matmul(wte, target_vec)
    norms = np.linalg.norm(wte, axis=1) * np.linalg.norm(target_vec)
    similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms!=0)
    nearest_indices = np.argsort(similarities)[::-1][:top_k]
    return nearest_indices, similarities[nearest_indices]

def main():
    model_id = "openai-community/gpt2"
    if not os.path.exists(f"weights/{model_id}"):
        print(f"Error: weights/{model_id} が見つかりません。")
        print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
        sys.exit(1)

    print("--- Embedding（埋め込み）体験 ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)
    tokenizer = Tokenizer(model_id)
    wte = params.wte  # (50257, 768)
    wpe = params.wpe  # (1024, 768)

    def get_vec(text):
        ids = tokenizer.encode(text)
        if not ids:
            return None
        return np.mean(wte[ids], axis=0)

    # 1. WTE: トークン埋め込み
    print("\n" + "=" * 50)
    print("1. WTE（トークン埋め込み）: 単語間のコサイン類似度")
    pairs = [("cat", "dog"), ("cat", "apple"), ("king", "queen"), ("man", "woman")]
    for w1, w2 in pairs:
        v1, v2 = get_vec(w1), get_vec(w2)
        sim = cosine_similarity(v1, v2)
        print(f"  {w1:8} <-> {w2:8} : {sim:.4f}")

    # 2. 最近傍探索
    print("\n" + "=" * 50)
    print("2. 最近傍探索: 'king' に近いトークン")
    v_king = get_vec("king")
    indices, sims = find_nearest(v_king, wte, top_k=6)
    for idx, sim in zip(indices, sims):
        token = tokenizer.decode([int(idx)])
        print(f"  {token:10} (ID: {idx:5}, 類似度: {sim:.4f})")

    # 3. ベクトル演算
    print("\n" + "=" * 50)
    print("3. ベクトル演算: 'king - man + woman'")
    v_result = get_vec("king") - get_vec("man") + get_vec("woman")
    indices, sims = find_nearest(v_result, wte, top_k=5)
    for idx, sim in zip(indices, sims):
        token = tokenizer.decode([int(idx)])
        print(f"  {token:10} (ID: {idx:5}, 類似度: {sim:.4f})")

    # 4. WPE: 位置埋め込み
    print("\n" + "=" * 50)
    print("4. WPE（位置埋め込み）: 位置間のコサイン類似度")
    print("  位置0を基準に、各位置との類似度:")
    positions = [0, 1, 2, 5, 10, 50, 100, 500]
    for pos in positions:
        sim = cosine_similarity(wpe[0], wpe[pos])
        print(f"  位置 0 <-> 位置 {pos:3d} : {sim:.4f}")

    # 5. Embedding の合成
    print("\n" + "=" * 50)
    print("5. 埋め込みの合成: WTE + WPE")
    text = "The capital of France is"
    input_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([i]) for i in input_ids]
    x = wte[input_ids] + wpe[np.arange(len(input_ids))]
    print(f"  入力: '{text}'")
    print(f"  トークン: {tokens}")
    print(f"  WTE 形状: {wte[input_ids].shape}")
    print(f"  WPE 形状: {wpe[np.arange(len(input_ids))].shape}")
    print(f"  合成後の形状: {x.shape}")

    print("\n--- 体験終了 ---")

if __name__ == "__main__":
    main()
