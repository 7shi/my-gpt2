import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import TransformerBlock
import os
import sys

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main():
    model_id = "openai-community/gpt2"
    if not os.path.exists(f"weights/{model_id}"):
        print(f"Error: weights/{model_id} が見つかりません。")
        print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
        sys.exit(1)

    print("--- 残差接続と Transformer Block 体験 ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)
    tokenizer = Tokenizer(model_id)
    n_head = 12

    # 入力テキスト
    text = "Machine Learning"
    input_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([i]) for i in input_ids]

    # Embedding
    x = params.wte[input_ids] + params.wpe[np.arange(len(input_ids))]
    x = x[np.newaxis, ...]

    print(f"\n入力: '{text}' (トークン数: {len(tokens)})")

    # 1. 1ブロック内の処理ステップ
    print("\n" + "=" * 50)
    print("1. Block 0 の内部処理ステップ")
    block = params.blocks[0]

    x_input = x.copy()
    pos = -1  # 最後のトークンのベクトルで観察
    print(f"  (最後のトークン '{tokens[-1].strip()}' の768次元ベクトルの標準偏差)")
    print()
    print(f"|ステップ | 標準偏差 | 備考 |")
    print(f"|---|---|---|")

    print(f"|入力 | {np.std(x_input[0, pos]):.4f} | |")

    x_ln1 = block.ln_1(x_input)
    print(f"|LayerNorm 1 後 | {np.std(x_ln1[0, pos]):.4f} | |")

    x_attn = block.attn(x_ln1, n_head=n_head)
    print(f"|Attention 出力 | {np.std(x_attn[0, pos]):.4f} | |")

    x_res1 = x_input + x_attn
    print(f"|残差接続 1 後 | {np.std(x_res1[0, pos]):.4f} | 入力 + Attention |")

    x_ln2 = block.ln_2(x_res1)
    print(f"|LayerNorm 2 後 | {np.std(x_ln2[0, pos]):.4f} | |")

    x_mlp = block.mlp(x_ln2)
    print(f"|MLP 出力 | {np.std(x_mlp[0, pos]):.4f} | |")

    x_res2 = x_res1 + x_mlp
    print(f"|残差接続 2 後 | {np.std(x_res2[0, pos]):.4f} | 残差1 + MLP |")

    # 2. 12ブロック全体の変化
    print("\n" + "=" * 50)
    print("2. 12ブロックを通した表現の変化")
    x_layer = x.copy()
    print(f"  (最後のトークン '{tokens[-1].strip()}' の768次元ベクトルで観察)")
    print()
    print(f"|層 | 標準偏差 | Emb からの cos 類似度 |")
    print(f"|---|---|---|")
    emb_vec = x[0, -1].copy()  # 最後のトークンの Embedding ベクトル
    print(f"|Emb | {np.std(x_layer[0, -1]):.4f} | {1.0:.4f} |")
    for i, block_params in enumerate(params.blocks):
        block_obj = TransformerBlock(block_params, n_head)
        x_layer = block_obj(x_layer)
        sim = cosine_similarity(emb_vec, x_layer[0, -1])
        print(f"|{i} | {np.std(x_layer[0, -1]):.4f} | {sim:.4f} |")
    x_ln_f = params.ln_f(x_layer)
    sim = cosine_similarity(emb_vec, x_ln_f[0, -1])
    print(f"|ln_f | {np.std(x_ln_f[0, -1]):.4f} | {sim:.4f} |")

    # 3. 文脈による表現の変化
    print("\n" + "=" * 50)
    print("3. 文脈による表現の変化")
    texts = [
        "The river bank was covered",
        "The money bank was covered",
    ]
    target_word = " bank"
    target_id = tokenizer.encode(target_word)[0]

    vecs_by_layer = []
    for t in texts:
        ids = tokenizer.encode(t)
        bank_pos = ids.index(target_id)
        xi = params.wte[ids] + params.wpe[np.arange(len(ids))]
        xi = xi[np.newaxis, ...]

        layer_vecs = [xi[0, bank_pos].copy()]
        for bp in params.blocks:
            blk = TransformerBlock(bp, n_head)
            xi = blk(xi)
            layer_vecs.append(xi[0, bank_pos].copy())
        xi_ln = params.ln_f(xi)
        layer_vecs.append(xi_ln[0, bank_pos].copy())
        vecs_by_layer.append(layer_vecs)

    print(f"  文A: '{texts[0]}'")
    print(f"  文B: '{texts[1]}'")
    print(f"  比較対象: '{target_word}' のベクトル")
    print()
    print(f"|層 | 文A-文B cos 類似度 |")
    print(f"|---|---|")
    for layer_idx in range(14):
        sim = cosine_similarity(vecs_by_layer[0][layer_idx], vecs_by_layer[1][layer_idx])
        if layer_idx == 0:
            label = "Emb"
        elif layer_idx == 13:
            label = "ln_f"
        else:
            label = str(layer_idx - 1)
        print(f"|{label} | {sim:.4f} |")

    # 4. 文章レベルの埋め込み
    print("\n" + "=" * 50)
    print("4. 文章レベルの埋め込み")

    sentences = [
        "The cat sat on the mat.",
        "A kitten was resting on the rug.",
        "Dogs are loyal and friendly animals.",
        "The stock market crashed yesterday.",
        "Investors lost money in the financial crisis.",
        "She enjoys reading books before bed.",
        "He likes to read novels at night.",
        "The weather is sunny and warm today.",
        "It is raining heavily outside.",
        "Python is a popular programming language.",
    ]

    def get_sentence_vector(text, use_ln_f=False):
        ids = tokenizer.encode(text)
        xi = params.wte[ids] + params.wpe[np.arange(len(ids))]
        xi = xi[np.newaxis, ...]
        for bp in params.blocks:
            blk = TransformerBlock(bp, n_head)
            xi = blk(xi)
        if use_ln_f:
            xi = params.ln_f(xi)
        return xi[0, -1].copy()

    def show_ranking(results):
        for rank, (sim, s) in enumerate(results, 1):
            print(f"  {rank:2d}. {sim:7.4f}  {s}")

    def show_similarity(sent_vecs, label="", query_idx=0):
        query_text, query_vec = sent_vecs[query_idx]
        label_str = f"（{label}）" if label else ""
        print(f"\n  基準文: '{query_text}'{label_str}")
        results = []
        for i, (s, v) in enumerate(sent_vecs):
            if i == query_idx:
                continue
            sim = cosine_similarity(query_vec, v)
            results.append((sim, s))
        results.sort(reverse=True)
        show_ranking(results)

    def show_keyword_search(sent_vecs, keywords, use_ln_f=False):
        label = "LayerNorm あり" if use_ln_f else "LayerNorm なし"
        for kw in keywords:
            kw_vec = get_sentence_vector(kw, use_ln_f=use_ln_f)
            print(f"\n  キーワード: '{kw}'（{label}）")
            results = []
            for s, v in sent_vecs:
                sim = cosine_similarity(kw_vec, v)
                results.append((sim, s))
            results.sort(reverse=True)
            show_ranking(results)

    keywords = ["animal", "finance", "programming"]

    # LayerNorm なし
    print("\n  --- LayerNorm なし ---")
    print("  文章ベクトルを計算中...")
    sent_vecs = [(s, get_sentence_vector(s)) for s in sentences]

    print("\n  [文章間の類似度]")
    show_similarity(sent_vecs, "LayerNorm なし")

    print(f"\n  [キーワード検索]")
    show_keyword_search(sent_vecs, keywords)

    # LayerNorm あり
    print("\n  --- LayerNorm あり ---")
    print("  文章ベクトルを計算中...")
    sent_vecs_ln = [(s, get_sentence_vector(s, use_ln_f=True)) for s in sentences]

    print("\n  [文章間の類似度]")
    show_similarity(sent_vecs_ln, "LayerNorm あり")

    print(f"\n  [キーワード検索]")
    show_keyword_search(sent_vecs_ln, keywords, use_ln_f=True)

    print("\n--- 体験終了 ---")

if __name__ == "__main__":
    main()
