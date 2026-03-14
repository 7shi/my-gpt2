import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import layer_norm, TransformerBlock
import os
import sys

def get_sentence_embedding(text, model_params):
    """
    文章を入力し、最終層の『最後のトークン』のベクトルを返す。
    これが文章全体（それまでの全ての単語）の情報を濃縮したベクトルになる。
    """
    tokenizer = Tokenizer("openai-community/gpt2")
    input_ids = tokenizer.encode(text)
    
    # 1. Embedding 層
    x = model_params.wte[input_ids] + model_params.wpe[np.arange(len(input_ids))]
    x = x[np.newaxis, ...] # (1, seq_len, 768)

    # 2. 全 12 層の Block を通過
    n_head = 12
    for p in model_params.blocks:
        block = TransformerBlock(p, n_head)
        x = block(x)

    # 3. 最終 LayerNorm
    x = layer_norm(x, model_params.ln_f)
    
    # 最後のトークンのベクトルを返す (1, seq_len, 768) -> (768,)
    return x[0, -1, :]

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    model_id = "openai-community/gpt2"
    if not os.path.exists(f"weights/{model_id}"):
        print(f"Error: weights/{model_id} が見つかりません。")
        sys.exit(1)

    print("--- 文章全体の埋め込み (Sentence Embedding) 体験 ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)

    # 3つの文を比較
    # A と B は似た構造だが、意味が異なる。C は全く異なる内容。
    text_a = "The bank of the river."
    text_b = "The bank of the country."
    text_c = "Apple is a tech company."

    print(f"\n文 A: '{text_a}'")
    print(f"文 B: '{text_b}'")
    print(f"文 C: '{text_c}'")
    print("-" * 40)

    # 各文の「文章全体」のベクトルを取得
    print("文章ベクトルを計算中...")
    vec_a = get_sentence_embedding(text_a, params)
    vec_b = get_sentence_embedding(text_b, params)
    vec_c = get_sentence_embedding(text_c, params)

    print("\n[文章ベクトル（最後のトークン）の類似度比較]")
    
    sim_ab = cosine_similarity(vec_a, vec_b)
    print(f"  文 A <-> 文 B (bank) : {sim_ab:.4f}")

    sim_ac = cosine_similarity(vec_a, vec_c)
    print(f"  文 A <-> 文 C (全然違う) : {sim_ac:.4f}")

    sim_bc = cosine_similarity(vec_b, vec_c)
    print(f"  文 B <-> 文 C (全然違う) : {sim_bc:.4f}")

    print("\n[考察]")
    print("1. 文章全体のベクトルを比較すると、内容の近さが数値で現れます。")
    print("2. 文 A と B は単語の多くが共通していますが、意味の差異 (river vs country) をモデルが捉えています。")
    print("3. 全く異なる内容の 文 C との類似度は、さらに低くなる傾向があります。")
    print("4. これが、RAG において『意味の近いドキュメントを探す』ための基礎技術です。")

    print("\n--- 体験終了 ---")

if __name__ == "__main__":
    main()
