import numpy as np
from my_gpt2.tokenizer import Tokenizer
from my_gpt2.loader import load_gpt2_weights
from my_gpt2.model import GPT2, softmax
import os
import sys

def get_top_tokens(logits, tokenizer, top_k=5):
    """
    ロジットから上位 k 個のトークンとその確率を抽出して返す。
    """
    probs = softmax(logits)
    top_indices = np.argsort(probs)[::-1][:top_k]
    return [(tokenizer.decode([int(idx)]), probs[idx]) for idx in top_indices]

def main():
    model_id = "openai-community/gpt2"
    if not os.path.exists(f"weights/{model_id}"):
        print(f"Error: weights/{model_id} が見つかりません。")
        sys.exit(1)

    print("--- テキスト生成 (Generation) 体験 ---")
    print("重みをロード中...")
    params = load_gpt2_weights(model_id)
    tokenizer = Tokenizer(model_id)
    model = GPT2(params, n_head=12)
    
    # 始まりの言葉 (Prompt)
    prompt = "Artificial Intelligence will"
    input_ids = tokenizer.encode(prompt)
    
    print(f"\nプロンプト: '{prompt}'")
    print("-" * 40)

    # 現在の入力を保持するリスト
    current_ids = input_ids.copy()

    # 3トークン分だけ生成の様子を詳しく観察する
    for step in range(3):
        print(f"\nStep {step + 1}: 次のトークンを予測中...")
        
        # モデルに入力 (全コンテキストを渡す)
        # logits: (1, seq_len, vocab_size)
        logits = model(np.array([current_ids]))
        
        # 最後のトークンのロジット（次の単語の予測）を取り出す
        next_token_logits = logits[0, -1, :]
        
        # 候補を表示
        top_tokens = get_top_tokens(next_token_logits, tokenizer)
        print(f"  上位候補:")
        for i, (token, prob) in enumerate(top_tokens):
            print(f"    {i+1}. '{token:12}' : {prob:.4f}")
        
        # 今回は最も確率が高いもの (Greedy) を選ぶ
        next_token = int(np.argmax(next_token_logits))
        current_ids.append(next_token)
        
        print(f"  -> 選択されたトークン: '{tokenizer.decode([next_token])}'")

    print("-" * 40)
    final_text = tokenizer.decode(current_ids)
    print(f"\n最終的な生成文: '{final_text}'")

    print("\n[考察]")
    print("1. GPT-2 は『これまでの全ての単語』をヒントに、次の 1 単語を予測します。")
    print("2. 上位候補を見ると、どれも文法的に自然な続きになっていることが分かります。")
    print("3. この予測と選択を繰り返すことで（自己回帰）、長い文章が生まれます。")

    print("\n--- 体験終了 ---")
    print("これで GPT-2 の仕組み（埋め込み -> Attention -> MLP -> 生成）の旅は完了です！")

if __name__ == "__main__":
    main()
