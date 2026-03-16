import regex
from my_gpt2.tokenizer import Tokenizer, bytes_to_unicode, get_pairs
import os
import sys

model_id = "openai-community/gpt2"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-gpt2' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- BPE トークナイザー ---")
tokenizer = Tokenizer(model_id)

# 1. 事前分割（Pre-tokenization）
print("\n" + "=" * 50)
print("1. 事前分割（Pre-tokenization）")
text = "Hello, world! It's a test."
chunks = regex.findall(tokenizer.pat, text)
print(f"  入力: {repr(text)}")
print(f"  分割: {chunks}")

# 2. bytes_to_unicode マッピング
print("\n" + "=" * 50)
print("2. bytes_to_unicode マッピング（代表例）")
b2u = bytes_to_unicode()
examples = [
    (0x00, "制御文字"),
    (0x0a, "改行"),
    (0x20, "space"),
    (0x21, "!"),
    (0x41, "A"),
    (0x7F, "制御文字"),
    (0xC4, "0xC4（'Ġ'の1バイト目）"),
    (0xA0, "0xA0（'Ġ'の2バイト目）"),
]
for b, desc in examples:
    u = b2u[b]
    print(f"  0x{b:02X} ({desc:20s}) -> {repr(u)}  (U+{ord(u):04X})")

# 3. BPEマージ過程（"Hello" の例）
print("\n" + "=" * 50)
print("3. BPEマージ過程（\"Hello\" の例）")
token = "Hello"
# byte_encoder 変換後（ASCII のためそのまま）
token_encoded = "".join(tokenizer.byte_encoder[b] for b in token.encode("utf-8"))
print(f"  byte_encoder 変換後: {repr(token_encoded)}")
word = tuple(token_encoded)
print(f"  初期: {list(word)}")

step = 1
while True:
    pairs = get_pairs(word)
    if not pairs:
        break
    bigram = min(pairs, key=lambda pair: tokenizer.bpe_ranks.get(pair, float("inf")))
    if bigram not in tokenizer.bpe_ranks:
        break
    rank = tokenizer.bpe_ranks[bigram]
    first, second = bigram
    new_word = []
    i = 0
    while i < len(word):
        try:
            j = word.index(first, i)
            new_word.extend(word[i:j])
            i = j
        except ValueError:
            new_word.extend(word[i:])
            break
        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
            new_word.append(first + second)
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    word = tuple(new_word)
    print(f"  step{step}: {list(bigram)} (rank={rank}) -> {list(word)}")
    step += 1
    if len(word) == 1:
        break

token_id = tokenizer.encoder[word[0]]
print(f"  結果: ID={token_id}")

# 4. エンコード例
print("\n" + "=" * 50)
print("4. エンコード例")
text = "Hello, world! It's a test."
chunks = regex.findall(tokenizer.pat, text)
chunks_encoded = ["".join(tokenizer.byte_encoder[b] for b in c.encode("utf-8")) for c in chunks]
ids = tokenizer.encode(text)
print(f"  入力: {repr(text)}")
print(f"  分割: {chunks}")
print(f"  変換: {chunks_encoded}")
print(f"  IDs:  {ids}")

# 5. デコード例
print("\n" + "=" * 50)
print("5. デコード例")
decoded = tokenizer.decode(ids)
print(f"  IDs:   {ids}")
print(f"  復元:  {repr(decoded)}")
print(f"  一致:  {decoded == text}")
