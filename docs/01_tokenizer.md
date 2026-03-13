# GPT-2 トークナイザー (Byte-Pair Encoding, BPE) 解説

GPT-2のトークナイザーは、人間が読む「文章」とモデルが計算できる「ID」の橋渡しをしています。
「バイト単位の処理」と「再帰的な結合ルール」を組み合わせた仕組みになっています。

---

### ステップ 1: バイトレベルのUnicodeマッピング

GPT-2は世界中のあらゆる文字を扱うため、最小単位を「バイト（0〜255）」として扱います。
しかし、制御文字や空白などは正規表現や表示で問題を起こすため、
「256種類のバイトを、見た目が安全な256種類のUnicode文字に1対1で変換する」というトリックを使います。

```python
@lru_cache()
def bytes_to_unicode():
    """
    256種類のバイト(0-255)を、表示可能な256個のUnicode文字にマッピングする。
    これにより、空白や制御文字も「普通の文字」としてBPE for GPT-2で処理できるようになる。
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
```

---

### ステップ 2: 正規表現による事前分割 (Pre-tokenization)

BPEを適用する前に、テキストを「単語の塊」に分割します。
これは、句読点と単語が混ざって結合されたり、スペースが消えたりするのを防ぐためです。

```python
# GPT-2専用の正規表現
# 短縮形 ('s, 't, 'reなど) や、単語、数字、記号を適切に切り分ける
self.pat = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", regex.IGNORECASE)
```

---

### ステップ 3: BPE結合アルゴリズム (Core Algorithm)

これがBPEの心臓部です。分割された各塊に対して、「結合ルール（merges.txt）」に従って、
隣り合うペアを1つの新しい記号に置き換えていく処理を、結合できなくなるまで繰り返します。

```python
def bpe(self, token):
    word = tuple(token)
    pairs = get_pairs(word) # 隣り合うペアをすべて抽出
    if not pairs:
        return token

    while True:
        # 現在のペアの中で、学習済みルール(merges.txt)で最も優先順位が高い(ランクが低い)ものを探す
        bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
        if bigram not in self.bpe_ranks:
            break # 結合できるルールがなくなったら終了
        
        # 見つけたペア(first, second)を1つに結合して新しいリストを作る
        # 例: ('f', 'o', 'x') -> ('fo', 'x') -> ('fox',)
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
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)
    return " ".join(word)
```

---

### ステップ 4: IDへのマッピング (Encode) と復元 (Decode)

最後に、結合された文字列を `vocab.json` に照らして数値（ID）に変換します。
デコードはその逆順で、IDを文字に戻し、Unicodeマッピングを解除して元のバイト列（UTF-8）に戻します。

```python
def encode(self, text):
    bpe_tokens = []
    for token in regex.findall(self.pat, text):
        # 1. バイト列をUnicodeマッピング文字に変換
        token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
        # 2. BPEを適用して結合し、語彙辞書からIDを取得
        bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
    return bpe_tokens

def decode(self, tokens):
    # 1. IDを文字列に戻す
    text = "".join([self.decoder[token] for token in tokens])
    # 2. Unicodeマッピングを解除して元のバイト列に戻し、UTF-8でデコード
    text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
    return text
```

---

### 設計の動機：なぜ「バイトレベル」なのか？

1. **未知語 (OOV) 問題の完全な解決**: 
   従来の単語単位のトークナイザーでは、辞書にない単語はすべて `[UNK]` (Unknown) となり、情報が失われていました。バイト単位に分解することで、どんなに複雑な多言語や絵文字でも、256種類のバイトの組み合わせで必ず表現できるようになります。
2. **語彙サイズの効率化**: 
   全ての文字（Unicode文字数は14万以上）を個別に登録すると、モデルの語彙行列が巨大になりすぎます。頻出する「バイトの塊」を結合していくBPEにより、約5万という適切な語彙サイズで、効率的な表現力を持たせています。

### 設計の動機：なぜ「Unicodeマッピング」が必要なのか？

BPEの処理中には、「空白」や「制御文字」も記号として扱いたいというニーズがあります。しかし、標準的な正規表現エンジンやデバッグ表示において、生のバイトデータ（特に制御コード）は予期せぬ挙動を引き起こします。
そこで、**「見た目が普通の文字だが、中身は特定のバイトを指している」**という状態を作ることで、文字列処理の堅牢性と実装の簡潔さを両立させています。
