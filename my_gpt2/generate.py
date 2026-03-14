import numpy as np
import argparse
from .tokenizer import Tokenizer
from .model import GPT2
from .loader import load_gpt2_weights
from .model import softmax

def generate(prompt, n_tokens_to_generate=30, temperature=1.0, model_id="openai-community/gpt2"):
    # 1. トークナイザーと重みを読み込む
    tokenizer = Tokenizer(model_id)
    params = load_gpt2_weights(model_id)

    # 2. モデルを初期化
    # GPT-2 small（124M）は12ヘッド
    model = GPT2(params, n_head=12)

    # 3. 入力をトークン化
    input_ids = tokenizer.encode(prompt)

    print(f"\nプロンプト: '{prompt}'")
    print(f"温度: {temperature}")
    print("生成中: ", end="", flush=True)

    # 4. 生成ループ
    # マルチバイト文字の途中で切れた場合のバッファ
    byte_buffer = bytearray()

    for _ in range(n_tokens_to_generate):
        # モデルの最大位置埋め込みを超えないようにする
        inputs = np.array([input_ids[-1024:]])

        # 順伝播
        logits = model(inputs)

        # 最後のトークンのロジットを取得
        next_token_logits = logits[0, -1, :]

        # 温度を適用
        if temperature > 0:
            # ロジットをスケールしてランダムサンプリング
            next_token_logits = next_token_logits / temperature
            probs = softmax(next_token_logits)
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            # 温度が0の場合は貪欲探索
            next_token = int(np.argmax(next_token_logits))

        # シーケンスに追加
        input_ids.append(next_token)

        # 新トークンの生のバイト列を取得
        token_str = tokenizer.decoder[next_token]
        token_bytes = bytes([tokenizer.byte_decoder[c] for c in token_str])
        byte_buffer.extend(token_bytes)

        # バッファをUTF-8としてデコードを試みる
        try:
            # バッファ全体が有効なUTF-8の場合
            decoded_text = byte_buffer.decode("utf-8")
            print(decoded_text, end="", flush=True)
            byte_buffer.clear()
        except UnicodeDecodeError as e:
            # 有効な部分だけデコード
            valid_bytes = byte_buffer[:e.start]
            if valid_bytes:
                print(valid_bytes.decode("utf-8"), end="", flush=True)
                # 無効な部分は次のトークンのために保持
                del byte_buffer[:e.start]

        # テキスト終端トークン（50256）で停止
        if next_token == 50256:
            print("\n[テキスト終端に到達]")
            break

    print("\n\n全出力:")
    # 最終デコード: 末尾の不完全なマルチバイト文字は無視する
    full_bytes = bytearray()
    for tid in input_ids:
        token_str = tokenizer.decoder[tid]
        full_bytes.extend([tokenizer.byte_decoder[c] for c in token_str])

    return full_bytes.decode("utf-8", errors="ignore")

def main():
    parser = argparse.ArgumentParser(
        description="NumPyによるGPT-2推論エンジン",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # プロンプトの位置引数（1語以上）
    parser.add_argument("prompt", nargs="+", help="生成を開始するプロンプトテキスト")
    parser.add_argument("-n", "--n_tokens", type=int, default=30, help="生成するトークン数")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="サンプリング温度（低いほど決定的）")
    parser.add_argument("-m", "--model", default="openai-community/gpt2", help="モデルID（例: openai-community/gpt2）")

    args = parser.parse_args()

    # プロンプトの単語をスペースで結合
    prompt_text = " ".join(args.prompt)

    output = generate(prompt_text, n_tokens_to_generate=args.n_tokens, temperature=args.temperature, model_id=args.model)
    print(output)

if __name__ == "__main__":
    main()
