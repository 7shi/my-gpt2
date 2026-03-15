# SentencePiece トークナイザー（ユニグラムモデル）解説

`rinna/japanese-gpt2-small` は BPE ではなく SentencePiece の**ユニグラムモデル**を使用します。
`my_gpt2/spiece.py` では、外部ライブラリなしに `spiece.model` を直接読み込んでエンコード・デコードを行います。

**ユニグラムモデル**とは、語彙内の各ピース（部分文字列）に「出現確率の対数（log probability）」をスコアとして持たせ、
テキストを「スコアの合計が最大になるピース列」に分割するアルゴリズムです。
「ユニグラム」とは各ピースを独立に扱う（前後のピースに依存しない）確率モデルであることを意味します。
最適分割の探索には動的計画法の一種である Viterbi アルゴリズムを用います（ステップ 3 で詳述）。

## BPE との違い

| 観点 | BPE（`openai-community/gpt2`） | ユニグラム（`rinna/japanese-gpt2-small`） |
|---|---|---|
| アルゴリズム | 最頻ペアを繰り返し結合 | 各ピースに対数確率スコアを持つ言語モデル |
| 分割方法 | 決定的（マージ順に従う） | 最尤分割（Viterbi で最高スコアを探す） |
| ファイル形式 | `vocab.json` + `merges.txt` | `spiece.model`（Protocol Buffers） |
| 単語境界 | スペースをバイトとして埋め込み | `▁`（U+2581）マーカーで表現 |

## ステップ 1: spiece.model のフォーマット（Protocol Buffers）

`spiece.model` は Protocol Buffers（protobuf）形式のバイナリファイルです。
`struct` モジュールだけで解析できます。

### フィールド構造：タグと値の繰り返し

protobuf のバイナリは「タグ + 値」の繰り返しです。タグは 1 バイト（または varint）で、
**フィールド番号**と**wire type**（値の型）を同時に表します。

```
tag = (field_num << 3) | wire_type
```

| wire type | 値 | 読み方 |
|---|---|---|
| 0 | varint | LEB128 可変長整数 |
| 2 | length-delimited | 長さ + データ（文字列・ネストしたメッセージ） |
| 5 | 32-bit | 4 バイト固定（float32 など） |

### 具体例：`<unk>` エントリ（先頭 16 バイト）

`spiece.model` の先頭を実際に読むと:

```
0a 0e 0a 05 3c 75 6e 6b 3e 15 00 00 00 00 18 02
```

これを 1 バイトずつ追うと次のようになります。

```
0a          → tag: field_num = 0x0a >> 3 = 1,  wire_type = 0x0a & 7 = 2
              → ModelProto.pieces（length-delimited, field 1）
0e          → length varint = 14（続くデータが 14 バイト）
  0a        → tag: field_num = 1, wire_type = 2 → SentencePiece.piece（文字列）
  05        → 文字列長 = 5
  3c 75 6e 6b 3e  → UTF-8: "<unk>"
  15        → tag: field_num = 2, wire_type = 5 → SentencePiece.score（float32）
  00 00 00 00     → 0.0f
  18        → tag: field_num = 3, wire_type = 0 → SentencePiece.type（varint）
  02        → 2 = UNKNOWN
```

### 具体例：`の` エントリ（ID=10）

```
0a 0a 0a 03 e3 81 ae 15 54 2c 6a c0
```

```
0a          → ModelProto.pieces（field 1, length-delimited）
0a          → length varint = 10
  0a        → SentencePiece.piece（field 1, 文字列）
  03        → 文字列長 = 3
  e3 81 ae  → UTF-8: "の"（3 バイト）
  15        → SentencePiece.score（field 2, float32）
  54 2c 6a c0  → -3.6590f（ユニグラムスコア）
              ※ type フィールドは省略 → デフォルト 1 (NORMAL)
```

### LEB128 varint の仕組み

varint は可変長整数で、各バイトの最上位ビット（MSB）が「次のバイトも続く」フラグです。
MSB=1 なら継続、MSB=0 なら終端。下位 7 ビットを低位から順に並べて値を作ります。

```
バイト列: 0x0e        → MSB=0（終端）、値 = 0x0e = 14
バイト列: 0x80 0x01   → 1バイト目 MSB=1（継続）: 0x00
                        2バイト目 MSB=0（終端）: 0x01
                        値 = 0x00 | (0x01 << 7) = 128
```

```python
def _read_varint(data, pos):
    """LEB128 形式の可変長整数をデコードする。"""
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift   # 下位 7 ビットを取り出して並べる
        if not (b & 0x80):              # MSB=0 なら終端
            break
        shift += 7
    return result, pos
```

### フィールドパーサー

タグを読んで wire type に応じた長さのデータを取り出し、`(field_num, wire_type, val)` を yield します。

```python
def _parse_fields(data, start, end):
    pos = start
    while pos < end:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:   # varint
            val, pos = _read_varint(data, pos)
            yield field_num, wire_type, val
        elif wire_type == 2: # length-delimited（文字列・ネストしたメッセージ）
            length, pos = _read_varint(data, pos)
            val = data[pos:pos + length]; pos += length
            yield field_num, wire_type, val
        elif wire_type == 5: # 32-bit float
            val = struct.unpack_from('<f', data, pos)[0]; pos += 4
            yield field_num, wire_type, val
        ...
```

wire_type=2 の `val` はネストしたメッセージの場合もあるため、
`_parse_fields(val, 0, len(val))` と再帰的に呼び出すことで `SentencePiece` メッセージ内部も解析できます。

### 設計の動機：なぜ Protocol Buffers を使うのか？

Protocol Buffers は Google が開発したバイナリシリアライズ形式です。
JSON より compact で、スキーマが変わってもフィールド番号が一致すれば読めるという前方互換性があります。
SentencePiece が protobuf を採用しているのは、この互換性と効率性のためです。

## ステップ 2: 語彙と正規化設定の読み込み

`spiece.model` の `ModelProto` には語彙以外にも正規化設定が含まれます。

```
ModelProto
  field 1（繰り返し）: SentencePiece（語彙エントリ）
    field 1: piece  （string, wire_type=2）
    field 2: score  （float32, wire_type=5）
    field 3: type   （int32: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 6=BYTE）
  field 2: TrainerSpec（学習時の設定。推論には不使用）
    field 1（繰り返し）: input （string: 学習データのパス）
    field 2: model_prefix （string: モデル出力先のプレフィックス）
    field 4: vocab_size （int: 語彙サイズ）
    field 30（繰り返し）: control_symbols （string: 例 "[PAD]", "[CLS]"）
  field 3: NormalizerSpec（正規化設定）
    field 1: name   （string: 例 "nmt_nfkc"）
    field 2: precompiled_charsmap （バイナリ: 変換テーブル）
```

`rinna/japanese-gpt2-small` の `spiece.model` では `normalizer_spec.name = "nmt_nfkc"` が設定されています。
これは NFKC Unicode 正規化をベースとした変換ルールで、全角英数字・記号を半角に統一するなどの処理を行います。

```python
def _load_vocab(model_path):
    data = open(model_path, "rb").read()
    vocab = []
    normalizer_name = None
    for fnum, wtype, val in _parse_fields(data, 0, len(data)):
        if fnum == 1 and wtype == 2:  # ModelProto.pieces
            piece, score, ptype = None, 0.0, 1
            for f2, w2, v2 in _parse_fields(val, 0, len(val)):
                if f2 == 1 and w2 == 2:
                    piece = v2.decode("utf-8")
                elif f2 == 2 and w2 == 5:
                    score = v2  # float32
                elif f2 == 3 and w2 == 0:
                    ptype = v2
            vocab.append((piece, score, ptype))
        elif fnum == 3 and wtype == 2:  # normalizer_spec
            for f2, w2, v2 in _parse_fields(val, 0, len(val)):
                if f2 == 1 and w2 == 2:  # normalizer_spec.name
                    normalizer_name = v2.decode("utf-8")
    return vocab, normalizer_name
```

`SentencePieceTokenizer.__init__` では、これを 3 つのテーブルと正規化名に変換します。

```python
vocab, self._normalizer = _load_vocab(path)
self._id_to_piece    = [piece for piece, score, ptype in vocab]
self._piece_to_id    = {piece: i for i, (piece, score, ptype) in enumerate(vocab)}
self._piece_to_score = {piece: score for piece, score, ptype in vocab}
```

### 設計の動機：なぜスコアを持つのか？

ユニグラムモデルではピース列全体の確率を各ピースの確率の積で表します。

```
P(▁, 日本語) = P(▁) × P(日本語)
```

確率は 0〜1 の小さな値なので、ピースが増えるほど積が極端に小さくなり浮動小数点数がアンダーフローします。
そこで対数を取ると、積が和に変わります。

```
log P(▁, 日本語) = log P(▁) + log P(日本語)
                 = (-3.5238) + (-9.9070)
                 = -13.4308
```

対数は単調増加なので大小関係は変わらず、確率の最大化がスコアの最大化と等価になります。
スコアは常に負の値で、0 に近いほど高確率なピースです（log(1) = 0）。

このスコアが Viterbi エンコードでの分割の「良さ」の基準になります。
高頻度な長いピース（「日本語」など）はスコアが高く、まとめて 1 トークンに分割されやすくなります。

## ステップ 3: ユニグラムモデルと Viterbi エンコード

テキスト `日本語` を分割する場合、`日|本|語`、`日本|語`、`日|本語` など多数の候補があります。
総当たりで全候補を列挙すると文字列長に対して指数的に増加するため、現実的ではありません。

Viterbi アルゴリズムはこれを**動的計画法（DP）**で効率的に解きます。
鍵となるアイデアは「位置 `i` までの最適分割が決まれば、それ以降の分割はその結果だけを引き継げばよい」という**最適部分構造**です。

まず SentencePiece の正規化として、スペースを `▁` に置換し先頭にも `▁` を付加します。
`▁` はセパレータではなくプレフィックスで、各単語の先頭に付きます。
`日本語 処理` → `▁日本語▁処理`（`▁処理` は `処理` の先頭に付く）。
先頭の `▁` は「文の先頭も単語の開始である」ことを示します。

```
テキスト（正規化後）: ▁日本語
```

`best[i]` として以下のデータを保持します。
- 位置 0〜i までを分割したときの最高スコア
- 開始インデックス（先行するbestのインデックスを兼ねます）
- そのスコアをもたらしたピース文字列（バックトラック時にトークン列を復元するために使う）

位置 `i` から `j` のピースが語彙にあれば、`best[j]` のスコアは `best[i].score + score(ピース)` となります。
これを左から右へ順に進めると、各位置で「そこまでの最善経路」が確定していきます。

例として `▁日本語`（n=4）を処理する場合の実際の更新を示します。
全位置を `-inf` のスコア（到達不能）で初期化し、`best[0] = (0.0, -1, None)` を起点とします。

i=0 から始まる候補を試みます（`▁日`・`▁日本語` は語彙にないためスキップ）。

```text
▁      (0→1): 0.0 + (-3.5238) = -3.5238  → 採用 best[1] = (-3.5238, 0, "▁")
▁日本  (0→3): 0.0 + (-9.2177) = -9.2177  → 採用 best[3] = (-9.2177, 0, "▁日本")
```

`best[1]` のスコア -3.5238 を起点に i=1 から始まる候補を試みます。

```text
日     (1→2): -3.5238 + (-5.9452) =  -9.4690 → 採用 best[2] = (-9.4690, 1, "日")
日本   (1→3): -3.5238 + (-7.4324) = -10.9562 → 棄却（既存の best[3] のスコアが高いため）
日本語 (1→4): -3.5238 + (-9.9070) = -13.4308 → 採用 best[4] = (-13.4308, 1, "日本語")
```

※ `best[3]` では `▁日本`（-9.2177）と `▁`+`日本`（-10.9562）が競合し、前者が勝ちました。

i=2、i=3 から始まる候補はいずれも既存の値に負けて棄却されます。

```text
本 (2→3): -9.4690 + (-7.4143) = -16.8833  → 棄却（既存の best[3] のスコアが高いため）
語 (3→4): -9.2177 + (-8.1796) = -17.3973  → 棄却（既存の best[4] のスコアが高いため）
```

※ `best[4]` では `▁`+`日本語`（-13.4308）と `▁日本`+`語`（-17.3973）が競合し、前者が勝ちました。

最終的な `best` の状態です。

```text
best[0] = (  0.0   , -1, None)
best[1] = ( -3.5238,  0, "▁")
best[2] = ( -9.4690,  1, "日")     ← バックトラックでは使われない
best[3] = ( -9.2177,  0, "▁日本")  ← バックトラックでは使われない
best[4] = (-13.4308,  1, "日本語")
```

文字をすべて消費した `best[4]` から開始インデックスを逆に辿る（バックトラック）と、最適なピース列が得られます。
- `best[4]` → `best[1]` → `best[0]`: `["▁", "日本語"]`


```python
def _normalize(self, text):
    """normalizer_spec.name に従ってテキストを正規化する。"""
    if self._normalizer and "nfkc" in self._normalizer:
        text = unicodedata.normalize("NFKC", text)
    return text

def encode(self, text):
    text = self._normalize(text)  # 例: ！→! ？→?
    normalized = "▁" + text.replace(" ", "▁")
    n = len(normalized)

    # best[i] = (累積スコア, 前の pos, piece)
    best = [(-math.inf, -1, None)] * (n + 1)
    best[0] = (0.0, -1, None)

    for i in range(n):
        if best[i][0] == -math.inf:
            continue
        for j in range(i + 1, n + 1):
            piece = normalized[i:j]
            if piece in self._piece_to_score:
                score = best[i][0] + self._piece_to_score[piece]
                if score > best[j][0]:
                    best[j] = (score, i, piece)
            elif j == i + 1:
                # 単一文字で語彙外の場合は UNK として扱う
                score = best[i][0] + (-1e10)
                if score > best[j][0]:
                    best[j] = (score, i, "<unk>")

    # バックトラックで最適ピース列を復元
    pieces = []
    pos = n
    while pos > 0:
        _, prev, piece = best[pos]
        pieces.append(piece)
        pos = prev
    pieces.reverse()

    return [self._piece_to_id.get(p, self.unk_id) for p in pieces]
```

単一文字でも語彙にない場合は `-1e10` のペナルティを付けた UNK (unknown) として進めます。

### 設計の動機：なぜ BPE ではなくユニグラムモデルなのか？

BPE は学習データの頻度に基づいて貪欲にペアを結合するため、「1 種類の分割」しか生成できません。
ユニグラムモデルは確率的な言語モデルとして定義されており、
「同じ文字列でも文脈に応じて最適な分割が異なる」ようなケースで有利です。
また、学習時に不要なピースを枝刈りして語彙を圧縮できるという利点もあります。

日本語は単語境界が曖昧なため、ユニグラムモデルの柔軟性が特に効果的です。

## ステップ 4: デコード（▁ の処理）

```python
def decode(self, tokens):
    pieces = [self._id_to_piece[i] for i in tokens]
    text = "".join(pieces)
    return text.replace("▁", " ").lstrip(" ")
```

ピース列を結合した後、`▁` をスペースに置換し、先頭のスペース（正規化で付けた `▁` 由来）を除去します。
BPE のようなバイトデコードは不要で、`▁` の置換だけで元のテキストが復元できます。

使用例:

```python
from my_gpt2.spiece import SentencePieceTokenizer

t = SentencePieceTokenizer("rinna/japanese-gpt2-small")
ids = t.encode("吾輩は猫である。")
print(ids)          # [9, 5361, 31082, 11, 4324, 27, 8]
print(t.decode(ids))  # '吾輩は猫である。'
```

## 付録: .model と .vocab の関係

`spiece.model` は推論に使うバイナリ（protobuf）です。内容を人間が読める形で確認したい場合は、語彙部分をタブ区切りテキストに変換した `.vocab` ファイルを使います。

```
ピース\tスコア
```

スコアはユニグラムモデルの対数確率で、値が大きい（0 に近い）ほど高頻度なピースです。特殊トークン（`<unk>` など）はスコア 0 で固定されています。

`.vocab` は語彙の閲覧用スナップショットであり、`spiece.model` が持つ全情報との 1:1 対応ではありません。
正規化設定（`normalizer_spec`）や学習時の設定（`trainer_spec`）は `.vocab` には含まれず、`spiece.model` からのみ読み取れます。

### model2vocab コマンド

`model2vocab` コマンドで変換できます。

```bash
uv run model2vocab weights/rinna/japanese-gpt2-small/spiece.model
```

出力先を変更する場合は `-o` で指定します。

```bash
uv run model2vocab weights/rinna/japanese-gpt2-small/spiece.model -o vocab.txt
```

`make download-rinna` 実行時に自動で変換され、`spiece.vocab` が生成されます。先頭 11 行:

```
<unk>	0.000000
<s>	0.000000
</s>	0.000000
[PAD]	0.000000
[CLS]	0.000000
[SEP]	0.000000
[MASK]	0.000000
、	-3.009356
。	-3.282608
▁	-3.523782
の	-3.658956
```
