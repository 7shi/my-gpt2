import math
import unicodedata
from my_gpt2.spiece import SentencePieceTokenizer
import os
import sys

def display_width(s):
    """東アジア文字（全角）を2カラム、それ以外を1カラムとして文字列の表示幅を返す。"""
    return sum(2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in s)

def ljust_display(s, width):
    return s + ' ' * max(0, width - display_width(s))

model_id = "rinna/japanese-gpt2-small"
if not os.path.exists(f"weights/{model_id}"):
    print(f"Error: weights/{model_id} が見つかりません。")
    print("先に 'make download-rinna' を実行して重みをダウンロードしてください。")
    sys.exit(1)

print("--- SentencePiece トークナイザー ---")
tokenizer = SentencePieceTokenizer(model_id)

# 1. 正規化
print("\n" + "=" * 50)
print("1. 正規化（Normalize）")
examples = ["日本語", "This is a pen.", "Ｈｅｌｌｏ　Ｗｏｒｌｄ"]
for t in examples:
    print(f"  {repr(t):30s} -> {repr(tokenizer._normalize(t))}")

# 2. spiece.model バイナリ検索（"日本語" エントリのオフセット）
print("\n" + "=" * 50)
print("2. spiece.model バイナリ検索（\"日本語\" エントリのオフセット）")
model_path = f"weights/{model_id}/spiece.model"
data = open(model_path, "rb").read()
target = "日本語".encode("utf-8")  # e6 97 a5 e6 9c ac e8 aa 9e
pos = data.find(target)
# エントリの先頭は文字列長フィールド(0x09)の2バイト前: タグ(0a) + 長さ(09) + 文字列
entry_start = pos - 2
print(f"  \"日本語\" UTF-8: {' '.join(f'{b:02x}' for b in target)}")
print(f"  文字列オフセット: 0x{pos:08X}")
# エントリ前後を含む16バイトを表示（ドキュメントのhexdumpと一致する範囲）
dump_start = entry_start - 2  # ModelProto field tag + length
dump_bytes = data[dump_start:dump_start + 18]
print(f"  {dump_start:08X}: {' '.join(f'{b:02x}' for b in dump_bytes)}")
# "日本語" エントリの ID を確認
piece_id = tokenizer._piece_to_id.get("日本語", -1)
score = tokenizer._piece_to_score.get("日本語", None)
print(f"  piece=\"日本語\", ID={piece_id}, score={score:.4f}")

# 3. Viterbi トレース（"日本語" の例）
print("\n" + "=" * 50)
print("3. Viterbiトレース（\"日本語\" の例）")
text_viterbi = "日本語"
normalized = tokenizer._normalize(text_viterbi)
print(f"  正規化後: {repr(normalized)}")
n = len(normalized)

best = [(-math.inf, -1, None)] * (n + 1)
best[0] = (0.0, -1, None)
print(f"  best[0] = (0.0, -1, None)  ← 起点")

for i in range(n):
    if best[i][0] == -math.inf:
        continue
    for j in range(i + 1, n + 1):
        piece = normalized[i:j]
        if piece in tokenizer._piece_to_score:
            score = best[i][0] + tokenizer._piece_to_score[piece]
            updated = score > best[j][0]
            label = "採用" if updated else "棄却"
            prev_score = best[j][0] if best[j][0] != -math.inf else None
            print(f"  {ljust_display(repr(piece), 12)} ({i}->{j}): {best[i][0]:.4f} + ({tokenizer._piece_to_score[piece]:.4f}) = {score:.4f}  → {label}", end="")
            if not updated and prev_score is not None:
                print(f"（既存 {prev_score:.4f}）", end="")
            print()
            if updated:
                best[j] = (score, i, piece)

print()
print("  最終 best[]:")
for idx, (sc, prev, piece) in enumerate(best):
    sc_str = f"{sc:.4f}" if sc != -math.inf else "-inf"
    print(f"    best[{idx}] = ({sc_str}, {prev}, {repr(piece)})")

# バックトラック
pieces = []
pos = n
while pos > 0:
    _, prev, piece = best[pos]
    pieces.append(piece)
    pos = prev
pieces.reverse()
print(f"\n  バックトラック結果: {pieces}")

# 4. エンコード例
print("\n" + "=" * 50)
print("4. エンコード例")
text_enc = "吾輩は猫である。"
ids = tokenizer.encode(text_enc)
print(f"  入力: {repr(text_enc)}")
print(f"  IDs:  {ids}")

# 5. デコード例
print("\n" + "=" * 50)
print("5. デコード例")
decoded = tokenizer.decode(ids)
print(f"  IDs:   {ids}")
print(f"  復元:  {repr(decoded)}")
print(f"  一致:  {decoded == text_enc}")
