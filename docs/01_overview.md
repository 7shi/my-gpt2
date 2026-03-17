ページ：**01** | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# GPT-2 推論パイプラインの全体像

## GPT-2 とは

GPT-2 は OpenAI が 2019 年 2 月に発表した言語モデルです。「テキストを受け取り、次の単語（トークン）を予測する」というシンプルな目的で訓練されていますが、その生成能力の高さから「危険すぎて公開できない」として大きな議論を呼びました。

当初は最小の 124M パラメーターモデルのみが公開され、最大の 1.5B パラメーターモデルは段階的リリースを経て 2019 年 11 月に全面公開されました。AI の責任ある公開という概念が広く議論される契機となったモデルでもあります。

現代の LLM（GPT-4, Claude など）も、Transformer アーキテクチャの基本は 2017 年の "Attention is All You Need" からほぼ変わっていません。GPT-2 との本質的な違いはパラメーター数と学習データの規模です。GPT-2 は現代の LLM の内部構造を理解するための最もコンパクトな出発点です。

### 日本語モデル: rinna/japanese-gpt2-small

2021 年、rinna 社（当時マイクロソフト子会社）が GPT-2 と同じ Transformer アーキテクチャを用い、日本語 Wikipedia と CC-100 で学習した日本語 GPT-2 モデルをオープンソースで公開しました。トークナイザーのみ SentencePiece（Unigram モデル）に置き換えられていますが、モデル本体の構造は GPT-2 と同一です。

本プロジェクトでは `-m rinna/japanese-gpt2-small` オプションで切り替えて使用できます。

## 推論パイプライン

GPT-2 に入力されたテキストは以下のパイプラインを通り、最終的に「次に来る単語の確率分布」として出力されます。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - [SentencePiece](03_spiece.md)
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

## トークナイザー: テキストからトークンIDへ

モデルに入力する前に、テキストをトークン ID 列に変換する必要があります。モデルに応じて 2 種類のトークナイザーを使い分けます。

### BPE

GPT-2 で使用します。バイトレベル BPE (Byte Pair Encoding) により、テキストをサブワード単位に分割します。

```python
# Step 0: トークナイザー — テキストをトークンIDの列に変換
tokenizer = Tokenizer("openai-community/gpt2")
input_ids = tokenizer.encode(text)
```

### SentencePiece

rinna の日本語モデルで使用します。Unigram モデルにより、日本語テキストを適切なサブワード単位に分割します。

```python
tokenizer = SentencePieceTokenizer("rinna/japanese-gpt2-small")
input_ids = tokenizer.encode(text)
```

## モデルの構造

モデルはいくつかの形式でアップロードされていますが、ここでは safetensors 形式を使用します。ファイルサイズは約 523 MB です。

```python
from safetensors.numpy import load_file
model_weights = load_file("model.safetensors")
```

`model_weights` はキーと値のペアで構成される辞書です。モデルの全パラメーターが行列やベクトルとして格納されています。768 は埋め込み次元、50257 は語彙数、1024 は最大トークン列長です。

| キー | 形状 | 役割 |
|---|---|---|
| `wte.weight` | (50257, 768) | Embedding（トークン） |
| `wpe.weight` | (1024, 768) | Embedding（位置） |
| `h.0.ln_1.weight\|bias` | (768,) | LayerNorm |
| `h.0.attn.c_attn.weight` | (768, 2304) | Attention（Q,K,V 結合） |
| `h.0.attn.c_attn.bias` | (2304,) | |
| `h.0.attn.c_proj.weight` | (768, 768) | Attention（出力射影） |
| `h.0.attn.c_proj.bias` | (768,) | |
| `h.0.ln_2.weight\|bias` | (768,) | LayerNorm |
| `h.0.mlp.c_fc.weight` | (768, 3072) | MLP（拡張） |
| `h.0.mlp.c_fc.bias` | (3072,) | |
| `h.0.mlp.c_proj.weight` | (3072, 768) | MLP（射影） |
| `h.0.mlp.c_proj.bias` | (768,) | |
| ... `h.1` 〜 `h.11` は同じ構造 | | |
| `ln_f.weight\|bias` | (768,) | 最終 LayerNorm |

キーの命名規則はモデルの階層構造をそのまま反映しています。

- Embedding: `wte`, `wpe`
- Transformer Block × 12: `h.0` 〜 `h.11`
  - LayerNorm: `ln_1`
  - Attention: `attn`
  - 残差接続: なし
  - LayerNorm: `ln_2`
  - MLP: `mlp`
  - 残差接続: なし
- 最終 LayerNorm: `ln_f`
- LM Head: `wte` と共用

残差接続は次に見るように足し算のため、パラメーターを持ちません。

## モデルの実装

推論パイプラインの中核部分のコードを示します。この中に主要な要素が凝縮されています。

```python
# Step 1: Embedding — トークンIDをベクトルに変換し、位置情報を加算
x = wte[input_ids] + wpe[np.arange(len(input_ids))]

# Step 2: Transformer Block × 12 — 文脈理解と特徴変換を繰り返す
for block in blocks:
    y = block.ln_1(x)  # 正規化
    y = block.attn(y)  # 文脈理解
    x = x + y          # 残差接続（足し算）
    z = block.ln_2(x)  # 正規化
    z = block.mlp(z)   # 特徴変換
    x = x + z          # 残差接続（足し算）

# Step 3: 最終 LayerNorm — 出力前の正規化
x = ln_f(x)

# Step 4: LM Head — 全語彙の埋め込みベクトルとの内積（ロジット）を一括計算
logits = x @ wte.T

# Step 5: サンプリング — 確率分布から次のトークンを選択
probs = softmax(logits[-1])  # 最後のロジットを 0～1 の確率に変換
next_id = np.argmax(probs)   # 貪欲法で最も確率の高いトークンを選択
```

`x` は実際には複数のベクトルを並べて行列の形式でまとめたものです。行列の形で計算すれば、全てのベクトルに対して同じ操作を一度に行うことができます。コメントでは分かりやすさを優先して、ベクトルとして説明しています。

`@` は行列の積ですが、ここでは全語彙の埋め込みベクトルとの内積をまとめて計算することを意図しています。

各ステップの詳細な仕組みは以降のドキュメントで順番に解説していきますが、以下ではデータの流れをざっくりと観察します。

## 実験：ステップごとに観察

入力テキストがトークナイザー (Step 0) からサンプリング (Step 5) までの各ステップをどのように通過するかを、形状・平均・標準偏差で追跡します。

**実行方法**: ([01_overview.py](01_overview.py))

```bash
uv run docs/01_overview.py
```

実行すると、入力テキスト "The capital of France is" を処理する様子を、ステップごとに追跡します。

### Step 0: トークナイザー

テキストをトークンIDの列に変換します。

```python
input_ids = tokenizer.encode(text)
```

```
  トークン: ['The', ' capital', ' of', ' France', ' is']
  トークンID: [464, 3139, 286, 4881, 318]
```

### Step 1: Embedding

5 つのトークンがそれぞれ 768 次元のベクトルに変換され、それを並べて行列の形式にまとめます。この時点ではまだ「文脈」を反映しておらず、各単語の静的な意味だけを持っています。

```python
x = wte[input_ids] + wpe[np.arange(len(input_ids))]
```

```
  形状: (5, 768)  (トークン数, 埋め込み次元)
  平均: -0.0039, 標準偏差: 0.2418
```

### Step 2: Transformer Block × 12

12 層のブロックで文脈の読み取り (Attention) と特徴変換 (MLP) を繰り返します。

```python
for block in blocks:
    y = block.ln_1(x)
    y = block.attn(y)
    x = x + y
    z = block.ln_2(x)
    z = block.mlp(z)
    x = x + z
```

| 層 | 平均 | 標準偏差 |
|---:|---:|---:|
| 0 | 0.0471 | 2.8242 |
| 1 | 0.1970 | 10.4311 |
| 2 | 0.8200 | 41.4148 |
| ... | | |
| 10 | 1.2198 | 50.9479 |
| 11 | 0.1078 | 14.2968 |

ブロックを通過するにつれ、標準偏差が急激に増加します。これは各層が文脈情報や特徴を書き込んでいることを示しています。最終層で標準偏差が減少するのは、出力に向けた調整が行われているためです。

### Step 3: 最終 LayerNorm

最終 LayerNorm (ln_f) でスケールが整えられます。

```python
x = ln_f(x)
```

```
  平均: 0.2684, 標準偏差: 6.8293
```

### Step 4: LM Head

行列の積により、全語彙の埋め込みベクトルとの内積を一括計算します。結果はロジットと呼ばれ、確率分布の前段階を表します。

```python
logits = x @ wte.T
```

```
  形状: (5, 50257)  (トークン数, 語彙数)
```

### Step 5: サンプリング

最終トークンのロジットを確率分布に変換して、それに基づいて次のトークンを選択します。`softmax` は数値を 0～1 の範囲に変換する関数です。

```python
probs = softmax(logits[-1])
```

```
  1. 確率 0.0846 ' the'
  2. 確率 0.0479 ' now'
  3. 確率 0.0462 ' a'
  4. 確率 0.0324 ' France'
  5. 確率 0.0322 ' Paris'
```

"Paris" が上位に入っており、モデルがフランスの首都についての知識を持っていることが分かります。

### 文章の生成

この予測を繰り返すことで文章を生成できます。生成したトークンを入力に追加し、再度パイプラインを実行する、という操作を繰り返す仕組みを**自己回帰生成**といいます。

各ステップで確率最大のトークンを選んだ結果を示します。この方式を**貪欲法** (greedy decoding) と呼び、決定論的でランダム性がないため結果は常に同じになります。

```python
next_id = np.argmax(probs)
```

> The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French

このように同じフレーズが繰り返されやすいのも貪欲法の特徴です。実用的な文章生成には、確率に基づいてトークンをランダムに選択する方法が利用されます。

---

ページ：**01** | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
