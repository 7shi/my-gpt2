import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
import os
import sys

def cosine_similarity(v1, v2):
    """2つのベクトルのコサイン類似度を計算する。"""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_nearest(target_vec, wte, top_k=5):
    """
    対象のベクトル target_vec に最も近いベクトルを wte 行列全体から探し出し、
    上位 top_k 個のトークン ID とその類似度を返す。
    """
    # 全ベクトルとのドット積を計算
    dot_products = np.matmul(wte, target_vec)
    # 各ベクトルのノルムで割ってコサイン類似度にする
    norms = np.linalg.norm(wte, axis=1) * np.linalg.norm(target_vec)
    # 0除算を防ぐ (norm が 0 のトークンがある場合に備えて)
    similarities = np.divide(dot_products, norms, out=np.zeros_like(dot_products), where=norms!=0)
    
    # 類似度が高い順に並べ替え
    nearest_indices = np.argsort(similarities)[::-1][:top_k]
    return nearest_indices, similarities[nearest_indices]

def main():
    # 重みの ID (標準的な GPT-2 small)
    model_id = "openai-community/gpt2"
    model_path = f"weights/{model_id}"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} が見つかりません。")
        print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
        sys.exit(1)

    print("--- 埋め込みモデル体験 (GPT-2 WTE) ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)
    wte = params.wte # Word Token Embedding (50257, 768)
    tokenizer = Tokenizer(model_id)

    def get_vec(text):
        ids = tokenizer.encode(text)
        if not ids:
            return None
        # 複数のトークンに分かれた場合は、平均を取ることで簡易的な文章ベクトルとする
        vecs = wte[ids]
        return np.mean(vecs, axis=0)

    print("ロード完了！\n")

    # 1. 類似度計算の例
    print("1. 単語間のコサイン類似度:")
    pairs = [("cat", "dog"), ("cat", "apple"), ("king", "queen"), ("man", "woman")]
    for w1, w2 in pairs:
        v1, v2 = get_vec(w1), get_vec(w2)
        sim = cosine_similarity(v1, v2)
        print(f"  {w1:8} <-> {w2:8} : {sim:.4f}")

    # 2. 最近傍探索の例
    print("\n2. 指定した単語に最も近い単語 (Nearest Neighbor):")
    target = "king"
    v_target = get_vec(target)
    indices, sims = find_nearest(v_target, wte, top_k=6)
    print(f"  '{target}' に近いトークン:")
    for idx, sim in zip(indices, sims):
        # 自分自身 (index 0) は飛ばさず表示してみる
        token = tokenizer.decode([int(idx)])
        print(f"    {token:10} (ID: {idx:5}, Sim: {sim:.4f})")

    # 3. ベクトル演算の例 (類推)
    print("\n3. ベクトル演算 (Analogy): 'king - man + woman'")
    v_king = get_vec("king")
    v_man = get_vec("man")
    v_woman = get_vec("woman")
    
    # 演算: 王 - 男 + 女
    v_result = v_king - v_man + v_woman
    indices, sims = find_nearest(v_result, wte, top_k=5)
    
    print("  'king - man + woman' の結果に近いトークン:")
    for idx, sim in zip(indices, sims):
        token = tokenizer.decode([int(idx)])
        print(f"    {token:10} (ID: {idx:5}, Sim: {sim:.4f})")

    print("\n--- 体験終了 ---")
    print("この 'wte' 行列が、GPT-2 の入り口（埋め込み）と出口（予測）の両方で使われています。")

if __name__ == "__main__":
    main()
