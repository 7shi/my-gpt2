ページ：[00](00_quickstart.md) | [01](01_overview.md) | [02](02_tokenizer.md) | **03** | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# SentencePiece トークナイザー（Unigram モデル）解説

`rinna/japanese-gpt2-small` は BPE ではなく SentencePiece の **Unigram モデル**を使用します。`my_gpt2/spiece.py` では、外部ライブラリなしに `spiece.model` を直接読み込んでエンコード・デコードを行います。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - **SentencePiece** ← この章
2. トークン ID 列
   - [Embedding](04_embedding.md)
3. ベクトル列
   - Transformer Block × 12
     - [LayerNorm](05_layer_norm.md)
     - [Attention](06_attention.md)
     - [残差接続](08_residual.md)
     - [LayerNorm](05_layer_norm.md)
     - [MLP](07_mlp.md)
     - [残差接続](08_residual.md)
   - [最終 LayerNorm](08_residual.md)
   - [LM Head](09_output.md)
4. ロジット
   - [サンプリング](09_output.md)
5. 次のトークン

## Unigram モデルとは？

**Unigram モデル**とは、語彙内の各ピース（部分文字列）に「出現確率の対数」をスコアとして持たせ、テキストを「スコアの合計が最大になるピース列」に分割するアルゴリズムです。「Unigram」とは各ピースを独立に扱う（前後のピースに依存しない）確率モデルであることを意味します。最適分割の探索には動的計画法の一種である Viterbi アルゴリズムを用います（ステップ 3 で詳述）。

### 設計の動機：なぜ出現確率の対数がスコアなのか？

Unigram モデルではピース列全体の確率を各ピースの確率の積で表します。

```text
P(▁, 日本語) = P(▁) × P(日本語)
```

確率は 0〜1 の小さな値なので、ピースが増えるほど積が極端に小さくなり浮動小数点数がアンダーフローします。そこで対数を取ると、積が和に変わります。

```text
log P(▁, 日本語) = log P(▁) + log P(日本語)
                 = (-3.5238) + (-9.9070)
                 = -13.4308
```

対数は単調増加なので大小関係は変わらず、確率の最大化がスコアの最大化と等価になります。スコアは常に負の値で、0 に近いほど高確率なピースです（log(1) = 0）。

このスコアが Viterbi エンコードでの分割の「良さ」の基準になります。高頻度な長いピース（「日本語」など）はスコアが高く、まとめて 1 トークンに分割されやすくなります。それにより、学習時に不要なピースを枝刈りして語彙を圧縮できるという利点があります。

### BPE との比較

| 観点 | BPE | Unigram |
|---|---|---|
| アルゴリズム | 最頻ペアを繰り返し結合 | 各ピースに対数確率スコアを持つ言語モデル |
| 分割方法 | 決定論的（マージ順に従う） | 最尤分割（Viterbi で最高スコアを探す） |
| ファイル形式 | `vocab.json` + `merges.txt` | `spiece.model`（Protocol Buffers） |
| 単語境界 | スペースをバイトとして埋め込み | `▁`（U+2581）マーカーで表現 |

## spiece.model のフォーマット（Protocol Buffers）

`spiece.model` は Protocol Buffers (protobuf) 形式のバイナリファイルです。

Protocol Buffers は Google が開発したバイナリシリアライズ形式です。JSON よりコンパクトで、フィールド名ではなくフィールド番号でデータを識別するため、スキーマにフィールドを追加・削除しても既存のデータを読み続けられる互換性があります。

### フィールド構造：タグと値の繰り返し

protobuf のバイナリは「タグ + 値」の繰り返しです。タグは 1 バイト（または varint）で、**フィールド番号**と**wire type**（値の型）を同時に表します。

```text
tag = (field_num << 3) | wire_type
```

| wire type | 値 | 読み方 |
|---|---|---|
| 0 | varint | LEB128 可変長整数 |
| 2 | length-delimited | 長さ + データ（文字列・ネストしたデータ） |
| 5 | 32-bit | 4 バイト固定（float32 など） |

### varint (LEB128 可変長整数)

varint は可変長整数で、各バイトの最上位ビット (MSB) が「次のバイトも続く」フラグです。MSB=1 なら継続、MSB=0 なら終端。下位 7 ビットを低位から順に並べて値を作ります。

```text
0x0e      → MSB=0（終端）、値 = 0x0e = 14
0x80 0x01 → 1バイト目 MSB=1（継続）: 0x00
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

### 具体例：UNKNOWN エントリ（先頭 16 バイト）

`spiece.model` の先頭を確認します。先頭の数字は16進オフセットです。

```text
00000000: 0a 0e 0a 05 3c 75 6e 6b 3e 15 00 00 00 00 18 02
```

フィールド構造を分析します。

```text
0a → tag: field_num = 0x0a >> 3 = 1,  wire_type = 0x0a & 7 = 2
   → field 1, length-delimited
0e → length varint = 14（続くデータが 14 バイト）
0a 05 3c 75 6e 6b 3e 15 00 00 00 00 18 02 → 14 バイトのデータ
```

14 バイトのデータはネストしたデータです。同じ要領で分析します。

```text
0a → field 1, length-delimited
05 → 文字列長 = 5
3c 75 6e 6b 3e → UTF-8: "<unk>"
15 → field 2, 32-bit (float32)
00 00 00 00 → 0.0f
18 → field 3, varint
02 → 2 (UNKNOWN)
```

ネストはこれら3つのフィールドがひとまとまりになっていることを表現しており、以下の構造に相当します。

```python
("<unk>", 0.0, UNKNOWN)
```

`<unk>` は UNKNOWN（未知語）を表す特別なトークンで、スコアは 0.0 です。

### スキーマとの対応

protobuf のデータは、スキーマを参照して初めて各フィールドの意味と構造が分かります。

- [sentencepiece_model.proto](https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto)

必要となる箇所を抜粋します。

```protobuf
message ModelProto {
  message SentencePiece {
    enum Type {
      NORMAL = 1;        // normal symbol
      UNKNOWN = 2;       // unknown symbol. only <unk> for now.
      CONTROL = 3;       // control symbols. </s>, <s>, <2ja> etc.
      USER_DEFINED = 4;  // user defined symbols.
                         // Typical usage of USER_DEFINED symbol
                         // is placeholder.
      BYTE = 6;          // byte symbols. Used when `byte_fallback` is true.
      UNUSED = 5;        // this piece is not used.
    }
    optional string piece = 1;  // piece must not be empty.
    optional float score = 2;
    optional Type type = 3 [default = NORMAL];

    // Customized extensions: the range of field numbers
    // are open to third-party extensions.
    extensions 200 to max;
  }

  // Sentence pieces with scores.
  repeated SentencePiece pieces = 1;
}
```

右辺の数字はフィールド番号を表します。field 1 の `pieces` は `SentencePiece` メッセージの繰り返しフィールドです。wire type 2 のデータが単純な文字列か埋め込みメッセージ（ネストしたデータ）かは、スキーマを見て初めて判断できます。

### 具体例："日本語" エントリ（ID=2481）

スキーマで構造が分かったところで、もう1つの例を見てみます。

```text
00008EF1: 0a 10 0a 09 e6 97 a5 e6 9c ac e8 aa 9e 15 2e 83 1e c1
```

```text
0a → ModelProto.pieces（field 1, length-delimited）
10 → length varint = 16
  0a → SentencePiece.piece（field 1, string）
  09 → 文字列長 = 9
  e6 97 a5 e6 9c ac e8 aa 9e → UTF-8: "日本語"
  15 → SentencePiece.score（field 2, float32）
  2e 83 1e c1 → -9.9070f
```

"日本語" は 9 バイトの UTF-8 文字列で、スコアは -9.9070 です。type フィールドが省略されているため、スキーマで定義されているデフォルト値 NORMAL が適用されます。

```python
SentencePiece(piece="日本語", score=-9.9070, type=NORMAL)
```

## ステップ 1: protobuf パーサー

タグを読んで wire type に応じた長さのデータを取り出し、`(field_num, wire_type, val)` を yield します。スキーマは参照しないため、ネストかどうかの判断は行いません。

```python
def _parse_fields(data, start, end):
    pos = start
    while pos < end:
        tag, pos = _read_varint(data, pos)
        field_num = tag >> 3
        wire_type = tag & 0x07
        if wire_type == 0:
            val, pos = _read_varint(data, pos)
            yield field_num, wire_type, val
        elif wire_type == 1:
            val = struct.unpack_from('<Q', data, pos)[0]; pos += 8
            yield field_num, wire_type, val
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            val = data[pos:pos + length]; pos += length
            yield field_num, wire_type, val
        elif wire_type == 5:
            val = struct.unpack_from('<f', data, pos)[0]; pos += 4
            yield field_num, wire_type, val
        else:
            raise ValueError(f"未対応の wire_type={wire_type} at pos={pos}")
```

浮動小数点数は 4 バイトの little-endian 形式で格納されているため、`struct.unpack_from('<f', ...)` でデコードします。

`wire_type=2` の `val` がネストしたメッセージであれば、`_parse_fields(val, 0, len(val))` として呼び出すことで `SentencePiece` メッセージ内部も解析できます。

## ステップ 2: 語彙と正規化設定の読み込み

`spiece.model` の `ModelProto` には語彙以外にも正規化設定が含まれます。

```proto
message NormalizerSpec {
  // name of normalization rule.
  optional string name = 1;
}

message ModelProto {
  // Spec for text normalization.
  optional NormalizerSpec normalizer_spec = 3;
}
```

`rinna/japanese-gpt2-small` の `spiece.model` では `normalizer_spec.name = "nmt_nfkc"` が設定されています。これは NFKC Unicode 正規化をベースとした変換ルールで、全角英数字・記号を半角に統一するなどの処理を行います。

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

## ステップ 3: Viterbi アルゴリズムとエンコード

テキスト `日本語` を分割する場合、`日|本|語`、`日本|語`、`日|本語` など多数の候補があります。総当たりで全候補を列挙すると文字列長に対して指数的に増加するため、現実的ではありません。

Viterbi アルゴリズムはこれを**動的計画法**（DP）で効率的に解きます。鍵となるアイデアは「位置 `i` までの最適分割が決まれば、それ以降の分割はその結果だけを引き継げばよい」という**最適部分構造**です。

まず `_normalize` でテキストを正規化します。正規化設定に応じた変換（全角英数字・記号を半角に統一する NFKC 変換など）の後、スペースを `▁` に置換し先頭にも `▁` を付加します。

例: "This is a pen." → "▁This▁is▁a▁pen."

`▁` はプレフィックスとして各単語の先頭に付きます。先頭の `▁` は「文の先頭も単語の開始である」ことを示します。

`best[i]` として以下のデータを保持します。

- 位置 0〜i までを分割したときの最高スコア
- 開始インデックス（先行するbestのインデックスを兼ねます）
- そのスコアをもたらしたピース文字列（バックトラック時にトークン列を復元するために使う）

位置 `i` から `j` のピースが語彙にあれば、`best[j]` のスコアは `best[i].score + score(ピース)` となります。これを左から右へ順に進めると、各位置で「そこまでの最善経路」が確定していきます。

例として `日本語` を処理する場合の実際の更新を示します。まずテキストを正規化して `▁日本語` とします。

`best[0] = (0.0, -1, None)` を起点とします。4文字のため、`best[1]` から `best[4]` までをスコア $-\infty$（到達不能）で初期化します。

i=0 から始まる候補を試みます。（`▁日` と `▁日本語` は語彙にありません）

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

i=2, i=3 から始まる候補はいずれも既存の値に負けて棄却されます。

```text
本 (2→3): -9.4690 + (-7.4143) = -16.8833  → 棄却（既存の best[3] のスコアが高いため）
語 (3→4): -9.2177 + (-8.1796) = -17.3973  → 棄却（既存の best[4] のスコアが高いため）
```

最終的な `best` の状態です。

```text
best[0] = (  0.0   , -1, None)
best[1] = ( -3.5238,  0, "▁")
best[2] = ( -9.4690,  1, "日")     ← バックトラックでは使われない
best[3] = ( -9.2177,  0, "▁日本")  ← バックトラックでは使われない
best[4] = (-13.4308,  1, "日本語")
```

末尾の `best[4]` から開始インデックスを逆に辿る（バックトラック）と、最適なピース列が得られます。

- `best[4]` → `best[1]` → `best[0]`: `["▁", "日本語"]`

```python
def _normalize(self, text):
    # normalizer_spec.name に応じたテキスト変換
    if self._normalizer and "nfkc" in self._normalizer:
        text = unicodedata.normalize("NFKC", text)  # 例: ！→! ？→?
    # スペースを ▁ に置換して先頭に ▁ を付加
    return "▁" + text.replace(" ", "▁")

def encode(self, text):
    normalized = self._normalize(text)
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

## ステップ 4: デコード

ピース列を結合した後、`▁` をスペースに置換し、先頭のスペース（正規化で付けた `▁` 由来）を除去します。BPE のようなバイトデコードは不要で、`▁` の置換だけで元のテキストが復元できます。

```python
def decode(self, tokens):
    pieces = [self._id_to_piece[i] for i in tokens]
    text = "".join(pieces)
    return text.replace("▁", " ").lstrip(" ")
```

使用例:

```python
from my_gpt2.spiece import SentencePieceTokenizer

t = SentencePieceTokenizer("rinna/japanese-gpt2-small")
ids = t.encode("吾輩は猫である。")
print(ids)          # [9, 5361, 31082, 11, 4324, 27, 8]
print(t.decode(ids))  # '吾輩は猫である。'
```

## 実験：SentencePiece トークナイザーの動作を確認

正規化・spiece.model バイナリ上の "日本語" エントリのオフセット検索・Viterbi トレース（各位置での採用/棄却とバックトラック）・encode/decode の一連の流れを確認します。実行結果は本文中で引用しています。

**実行方法**: ([03_spiece.py](03_spiece.py))

```bash
uv run docs/03_spiece.py
```

## 付録: .model と .vocab の関係

`spiece.model` は推論に使うバイナリ (protobuf) です。内容を人間が読める形で確認したい場合は、語彙部分をタブ区切りテキストに変換した `.vocab` ファイルを使います。

```
ピース\tスコア
```

スコアは Unigram モデルの対数確率で、値が大きい（0 に近い）ほど高頻度なピースです。特殊トークン（`<unk>` など）はスコア 0 で固定されています。

`.vocab` は語彙の閲覧用スナップショットであり、`spiece.model` が持つ全情報との 1:1 対応ではありません。正規化設定（`normalizer_spec`）や学習時の設定（`trainer_spec`）は `.vocab` には含まれず、`spiece.model` からのみ読み取れます。

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

---

ページ：[00](00_quickstart.md) | [01](01_overview.md) | [02](02_tokenizer.md) | **03** | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
