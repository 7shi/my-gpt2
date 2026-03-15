# GPT-2 トークナイザー (Byte-Pair Encoding, BPE) 解説

テキストをそのままモデルに入力することはできないため、まずトークナイザーで「トークン」と呼ばれる意味のある単位に分割し、それぞれに番号（ID）を割り当てます。

GPT-2は文字単位でも単語単位でもなく、「頻出するバイト列の塊」を学習によってトークンとして定義する BPE（Byte Pair Encoding）方式を採用しています。これにより、未知語をゼロにしながら、語彙を効率的なサイズに抑えることができます。

## ステップ 1: 正規表現による事前分割 (Pre-tokenization)

BPEを適用する前に、テキストを「単語の塊」に分割します。
これは、句読点と単語が混ざって結合されたり、スペースが消えたりするのを防ぐためです。

```python
# GPT-2専用の正規表現
# 短縮形 ('s, 't, 'reなど) や、単語、数字、記号を適切に切り分ける
self.pat = regex.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", regex.IGNORECASE)
```

たとえば `"Hello, world! It's a test."` という入力は次のように分割されます。

```text
入力: "Hello, world! It's a test."
分割: ['Hello', ',', ' world', '!', ' It', "'s", ' a', ' test', '.']
```

スペースは後続の単語に吸収される形（`' world'` のように先頭のスペースを含む）で扱われます。これがのちに `Ġ` として現れる元になります。

※ `\p{L}+` はUnicodeの「Letter」カテゴリに属する文字（アルファベット・漢字・ひらがななど）のみにマッチします。数字・記号・スペースは含まないため、`.+` とは異なります。正規表現パターン全体で `\p{L}`（文字）・`\p{N}`（数字）・`[^\s\p{L}\p{N}]`（記号）と役割分担しており、`"Hello, world!"` のような入力を適切に分割できます。

### 補足：分かち書きしない言語への影響

この正規表現パターンは英語を中心に設計されており、スペースで単語が区切られることを前提としています。日本語・中国語・タイ語など分かち書きしない言語では、`\p{L}+` が文全体を1チャンクとして扱うため、BPEが意味のある単位で分割できません。

たとえば `'日本語のテスト'` は1チャンクとして渡され、GPT-2の語彙に日本語の学習がほぼないため、バイト単位でバラバラに分解されます。

```text
事前分割: ['日本語のテスト']  ← 1チャンク
トークン:  ['æĹ', '¥', 'æľ', '¬', 'èª', 'ŀ', 'ãģ®', 'ãĥĨ', 'ãĤ¹ãĥĪ']  （9トークン）
```

7文字が9トークンになっており、英語の同程度の文字数と比べて非効率です。GPT-4で使われる tiktoken や LLaMA で使われる SentencePiece はこの問題を改善しています。

## ステップ 2: バイトレベルのUnicodeマッピング

各チャンクをUTF-8バイト列に変換し、最小単位を「バイト（0〜255）」として扱います。しかし、制御文字や空白などは正規表現や表示で問題を起こすため、「256種類のバイトを、見た目が安全な256種類のUnicode文字に1対1で変換する」というトリックを使います。

BPEの処理中には「空白」や「制御文字」も記号として扱いたいというニーズがありますが、生のバイトデータは正規表現エンジンやデバッグ表示で予期せぬ挙動を引き起こします。そこで「見た目が普通の文字だが、中身は特定のバイトを指している」という状態を作ることで、文字列処理の堅牢性と実装の簡潔さを両立させています。

```python
@lru_cache()
def bytes_to_unicode():
    """
    256種類のバイト(0-255)を、表示可能な256個のUnicode文字にマッピングする。
    これにより、空白や制御文字も「普通の文字」としてBPE for GPT-2で処理できるようになる。
    """
    non_printable = set([*range(0, 0x20 + 1), *range(0x7F, 0xA0 + 1), 0xAD])
    result = {}
    n = 0
    for b in range(256):
        if b in non_printable:
            result[b] = chr(256 + n)
            n += 1
        else:
            result[b] = chr(b)
    return result
```

表示可能な文字はそのまま保持し、それ以外の制御文字や空白はU+0100以降のUnicode文字に順番に割り当てます。

```text
0x00         -> 'Ā'   （制御文字のためU+0100にマッピング）
0x0a (改行)  -> 'Ċ'   （制御文字のためU+010Aにマッピング）
0x20 (space) -> 'Ġ'   （空白のためU+0120にマッピング）
0x21 (!)     -> '!'   （そのまま）
0x41 (A)     -> 'A'   （そのまま）
0x7F         -> 'ġ'   （制御文字のためU+0121にマッピング）
```

例えば日本語の「あ」はUTF-8で `0xE3 0x81 0x82` の3バイトになり、それぞれが別々のUnicode文字にマッピングされます。

- `0xE3` -> `'ã'`（表示可能文字のため自身にマッピング）
- `0x81` -> `'ģ'`（制御文字扱いのためU+0123にマッピング）
- `0x82` -> `'Ĥ'`（制御文字扱いのためU+0124にマッピング）

なお、スペース（0x20）の `'Ġ'` はBPE処理の内部でのみ使われる表現です。本来の `'Ġ'` はUTF-8では `0xC4 0xA0` の2バイトで、変換後は `'Äł'` になるため、混同されることはありません。

- `0xC4` -> `'Ä'` (そのまま)
- `0xA0` -> `'ł'` (空白のためU+0142にマッピング)

### 設計の動機：なぜ「バイトレベル」なのか？

1. **未知語 (OOV) 問題の完全な解決**:
   従来の単語単位のトークナイザーでは、辞書にない単語はすべて `[UNK]` (Unknown) となり、情報が失われていました。バイト単位に分解することで、どんなに複雑な多言語や絵文字でも、256種類のバイトの組み合わせで必ず表現できるようになります。
2. **語彙サイズの効率化**:
   全ての文字（Unicode文字数は14万以上）を個別に登録すると、モデルの語彙行列が巨大になりすぎます。頻出する「バイトの塊」を結合していくBPEにより、約5万という適切な語彙サイズで、効率的な表現力を持たせています。

## ステップ 3: BPE結合アルゴリズム (Core Algorithm)

分割された各チャンクに対して、「結合ルール（merges.txt）」に従って、隣り合うペアを1つの新しい記号に置き換えていく処理を、結合できなくなるまで繰り返します。BPEはチャンクをまたいで結合されることはなく、各チャンク内で独立して適用されます。

rank（ランク）とは、結合ルールの優先順位を示す整数です。`merges.txt` の上位にあるルールほど小さなランクを持ち、優先して適用されます。ランクが小さいほど、学習データ中でそのペアが多く現れたことを意味します。

※ rankは `merges.txt` の行番号（1始まり）から2を引いた値です。1行目はヘッダ（`#version: 0.2`）のためスキップされ、2行目が rank=0 になります。

`"Hello"` を例に、BPEマージの過程を追ってみます。

```text
初期: ['H', 'e', 'l', 'l', 'o']
step1: ['l', 'l'] (rank=41)      -> ['H', 'e', 'll', 'o']
step2: ['e', 'll'] (rank=439)    -> ['H', 'ell', 'o']
step3: ['ell', 'o'] (rank=10853) -> ['H', 'ello']
step4: ['H', 'ello'] (rank=15240) -> ['Hello']
結果: ID=15496
```

rankが小さいほど優先度が高く先にマージされます。最終的に `['Hello']` という1トークンになります。

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

## ステップ 4: IDへのマッピング (Encode) と復元 (Decode)

`vocab.json` はトークン文字列をIDにマッピングする辞書で、BPEで結合されたトークンがすべて登録されています。

```json
{ "!": 0, "\"": 1, "#": 2, "$": 3, ... }
```

トークン辞書に照らして数値（ID）に変換すれば、エンコードの完成です。

- "Hello" → ID 15496

ここまでの一連の流れを順番に適用することで、文字列からIDの列にエンコードできます。

```python
def encode(self, text):
    bpe_tokens = []
    for token in regex.findall(self.pat, text):
        # 1. バイト列をUnicodeマッピング文字に変換
        token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
        # 2. BPEを適用して結合し、語彙辞書からIDを取得
        bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
    return bpe_tokens
```

エンコードの例を示します。

```text
text: "Hello, world! It's a test."
分割: ['Hello', ',', ' world', '!', ' It', "'s", ' a', ' test', '.']
変換: ['Hello', ',', 'Ġworld', '!', 'ĠIt', "'s", 'Ġa', 'Ġtest', '.']
bpe_tokens: [15496, 11, 995, 0, 632, 338, 257, 1332, 13]
```

デコードはその逆で、IDを文字に戻し、Unicodeマッピングを解除して元のバイト列（UTF-8）に戻します。

```python
def decode(self, tokens):
    # 1. IDを文字列に戻す
    text = "".join([self.decoder[token] for token in tokens])
    # 2. Unicodeマッピングを解除して元のバイト列に戻し、UTF-8でデコード
    text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors="replace")
    return text
```

デコードの例を示します。

```text
tokens: [15496, 11, 995, 0, 632, 338, 257, 1332, 13]
text(1): "Hello,Ġworld!ĠIt'sĠaĠtest."
text(2): "Hello, world! It's a test."
```

### 補足：スペース付きトークンについて

GPT-2 では、スペースなしとスペース付きが別々のトークンとして登録されています。

```text
Hello  (ID 15496)  /  ĠHello (ID 18435)
world  (ID  6894)  /  Ġworld (ID   995)
It     (ID  1026)  /  ĠIt    (ID   632)
is     (ID   271)  /  Ġis    (ID   318)
```

文頭の単語はスペースなし、文中の単語はスペース付きトークンとして処理されます。エンコード後のトークン文字列に現れる `Ġ` はすべて「直前にスペースがある」ことを意味します。

この設計の主な目的はトークン数の削減です。スペースを独立したトークンにすると約40%トークン数が増えるため、スペースを後続の単語に吸収することで効率化しています。副産物として `'Hello'`（文頭）と `'ĠHello'`（文中）が別IDになり、モデルが位置情報を暗黙的に学習できるという利点もあります。一方で、語彙の多くがスペース付き・なしの対で占められる非効率もあり、後継モデルでは SentencePiece や tiktoken による改善が図られています。
