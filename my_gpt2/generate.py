import os
import numpy as np
import argparse
from .tokenizer import Tokenizer
from .spiece import SentencePieceTokenizer
from .model import GPT2
from .loader import load_gpt2_weights
from .model import softmax

def generate(prompt, n_tokens_to_generate=30, temperature=1.0, top_k=None, top_p=None, model_id="openai-community/gpt2", verbose=False, *, model=None, tokenizer=None):
    # 1. トークナイザーと重みを読み込む（未指定時）
    if tokenizer is None:
        spiece_path = f"weights/{model_id}/spiece.model"
        if os.path.exists(spiece_path):
            tokenizer = SentencePieceTokenizer(model_id)
        else:
            tokenizer = Tokenizer(model_id)
    if model is None:
        params = load_gpt2_weights(model_id, verbose=verbose)
        # GPT-2 small（124M）は12ヘッド
        model = GPT2(params, n_head=12)

    # 2. 入力をトークン化
    input_ids = tokenizer.encode(prompt)

    if verbose:
        print(f"\nプロンプト: '{prompt}'")
        print(f"温度: {temperature}, top_k: {top_k}, top_p: {top_p}")
        print("生成中: ", end="", flush=True)
    else:
        print(tokenizer.decode(input_ids), end="", flush=True)

    # 3. 生成ループ（KV キャッシュ使用）
    # マルチバイト文字の途中で切れた場合のバッファ
    byte_buffer = bytearray()
    kv_cache = None  # 初回は None（prefill）

    for _ in range(n_tokens_to_generate):
        if kv_cache is None:
            # Prefill: 全プロンプトを処理してキャッシュを構築
            inputs = np.array(input_ids[-1024:])
        else:
            # Incremental: 新トークンのみ処理
            inputs = np.array([input_ids[-1]])

        # 順伝播（KV キャッシュ付き）
        logits, kv_cache = model(inputs, kv_cache=kv_cache)

        # 1024 トークンを超えたらキャッシュを破棄して再構築
        if kv_cache[0][0].shape[2] >= 1024:
            kv_cache = None

        # 最後のトークンのロジットを取得
        next_token_logits = logits[-1, :]

        # 温度を適用
        if temperature > 0:
            # ロジットをスケールしてランダムサンプリング
            next_token_logits = next_token_logits / temperature
            if top_k is not None and top_k > 0:
                # 上位k個以外を -inf にマスク
                top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
                mask = np.full_like(next_token_logits, -np.inf)
                mask[top_k_indices] = next_token_logits[top_k_indices]
                next_token_logits = mask
            probs = softmax(next_token_logits)
            if top_p is not None and 0.0 < top_p < 1.0:
                # 確率の降順にソートし、累積確率が top_p を超えるまでのトークンのみ残す
                sorted_indices = np.argsort(probs)[::-1]
                cumulative_probs = np.cumsum(probs[sorted_indices])
                cutoff = np.searchsorted(cumulative_probs, top_p) + 1
                removed = sorted_indices[cutoff:]
                probs[removed] = 0.0
                probs /= probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            # 温度が0の場合は貪欲探索
            next_token = int(np.argmax(next_token_logits))

        # シーケンスに追加
        input_ids.append(next_token)

        # ストリーミング出力
        if isinstance(tokenizer, SentencePieceTokenizer):
            piece = tokenizer._id_to_piece[next_token]
            print(piece.replace("▁", " "), end="", flush=True)
        else:
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

        # テキスト終端トークンで停止
        if next_token == tokenizer.eos_id:
            if verbose:
                print("\n[テキスト終端に到達]")
            break

    print()

    if verbose:
        print("\n全出力:")

    if isinstance(tokenizer, SentencePieceTokenizer):
        return tokenizer.decode(input_ids)
    else:
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
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="サンプリング温度（低いほど決定論的）")
    parser.add_argument("-k", "--top_k", type=int, default=None, help="top-k サンプリング（上位k個のトークンから選択）")
    parser.add_argument("-p", "--top_p", type=float, default=None, help="top-p サンプリング（累積確率p以内のトークンから選択）")
    parser.add_argument("-m", "--model", default="openai-community/gpt2", help="モデルID（例: openai-community/gpt2）")
    parser.add_argument("-r", "--repeat", type=int, default=1, help="同じプロンプトを繰り返す回数")
    parser.add_argument("-s", "--seed", type=int, default=None, help="乱数シード（再現性のため）")
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細な情報を表示する")

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # モデルとトークナイザーを一度だけ読み込む
    spiece_path = f"weights/{args.model}/spiece.model"
    if os.path.exists(spiece_path):
        tokenizer = SentencePieceTokenizer(args.model)
    else:
        tokenizer = Tokenizer(args.model)
    params = load_gpt2_weights(args.model, verbose=args.verbose)
    model = GPT2(params, n_head=12)

    for prompt_text in args.prompt:
        for _ in range(args.repeat):
            output = generate(prompt_text, n_tokens_to_generate=args.n_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, model_id=args.model, verbose=args.verbose, model=model, tokenizer=tokenizer)
            if args.verbose:
                print(output)

if __name__ == "__main__":
    main()
