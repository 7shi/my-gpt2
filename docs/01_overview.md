ページ：**1** | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | [8](08_residual.md) | [9](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# GPT-2 推論パイプラインの全体像

## GPT-2 とは

GPT-2 は OpenAI が 2019 年 2 月に発表した言語モデルです。「テキストを受け取り、次の単語（トークン）を予測する」というシンプルな目的で訓練されていますが、その生成能力の高さから「危険すぎて公開できない」として大きな議論を呼びました。

当初は最小の 124M パラメータモデルのみが公開され、最大の 1.5B パラメータモデルは段階的リリースを経て 2019 年 11 月に全面公開されました。AI の責任ある公開という概念が広く議論される契機となったモデルでもあります。

現代の LLM（GPT-4, Claude など）も、Transformer アーキテクチャの基本は 2017 年の "Attention is All You Need" からほぼ変わっていません。GPT-2 との本質的な違いはパラメータ数と学習データの規模です。GPT-2 は現代の LLM の内部構造を理解するための最もコンパクトな出発点です。

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
tokenizer = Tokenizer("openai-community/gpt2")
input_ids = tokenizer.encode(text)
```

### SentencePiece

rinna の日本語モデルで使用します。Unigram モデルにより、日本語テキストを適切なサブワード単位に分割します。

```python
tokenizer = SentencePieceTokenizer("rinna/japanese-gpt2-small")
input_ids = tokenizer.encode(text)
```

## モデル実装

推論パイプラインの中核部分のコードを示します。この中に主要な要素が凝縮されています。

```python
# Step 1: Embedding — トークンIDをベクトルに変換し、位置情報を加算
x = wte[input_ids] + wpe[np.arange(len(input_ids))]

# Step 2: Transformer Block × 12 — 文脈理解と特徴変換を繰り返す
for block in blocks:
    x = block(x)

# Step 3: 最終 LayerNorm — 出力前の正規化
x = ln_f(x)

# Step 4: LM Head — ベクトルをロジット（確率分布の前段階）に変換
logits = x @ wte.T

# Step 5: サンプリング — 確率分布から次のトークンを選択
probs = softmax(logits[-1])
next_id = np.argmax(probs)  # greedy decoding
```

各ステップの詳細な仕組みは以降のドキュメントで順番に解説していきますが、以下ではデータの流れをざっくりと観察します。

## 実験：ステップごとに観察

入力テキストがトークナイザー (Step 0) からサンプリング (Step 5) までの各ステップをどのように通過するかを、形状・平均・標準偏差で追跡します。

**実行方法**: ([01_overview.py](01_overview.py))

```bash
uv run docs/01_overview.py
```

実行すると、入力テキスト `The capital of France is` を処理する様子を、ステップごとに追跡します。

### Step 0: トークナイザー

テキストをトークンIDの列に変換します。

```
  トークン: ['The', ' capital', ' of', ' France', ' is']
  トークンID: [464, 3139, 286, 4881, 318]
```

### Step 1: Embedding

5 つのトークンがそれぞれ 768 次元のベクトルに変換されます。この時点ではまだ「文脈」を反映しておらず、各単語の静的な意味だけを持っています。

```
  形状: (5, 768)  (トークン数, 埋め込み次元)
  平均: -0.0039, 標準偏差: 0.2418
```

### Step 2: Transformer Block × 12

12 層のブロックを通過するにつれ、標準偏差が急激に増加します。これは各層が文脈情報や特徴を書き込んでいることを示しています。最終層で標準偏差が減少するのは、出力に向けた調整が行われているためです。

| 層 | 平均 | 標準偏差 |
|---:|---:|---:|
| 0 | 0.0471 | 2.8242 |
| 1 | 0.1970 | 10.4311 |
| 2 | 0.8200 | 41.4148 |
| ... | | |
| 10 | 1.2198 | 50.9479 |
| 11 | 0.1078 | 14.2968 |

### Step 3: 最終 LayerNorm

最終 LayerNorm (ln_f) でスケールが整えられます。

```
  平均: 0.2684, 標準偏差: 6.8293
```

### Step 4: LM Head

全トークン位置に対して語彙サイズのロジット（確率分布の前段階）が得られます。

```
  形状: (5, 50257)  (トークン数, 語彙数)
```

### Step 5: サンプリング

最終トークンのロジットを確率分布に変換して、それに基づいて次のトークンを選択します。"Paris" が上位に入っており、モデルがフランスの首都についての知識を持っていることが分かります。

```
  1. 確率 0.0846 ' the'
  2. 確率 0.0479 ' now'
  3. 確率 0.0462 ' a'
  4. 確率 0.0324 ' France'
  5. 確率 0.0322 ' Paris'
```

この予測を繰り返すことで文章を生成できます。生成したトークンを入力に追加し、再度パイプラインを実行する、という操作を繰り返す仕組みを**自己回帰生成**といいます。

各ステップで確率最大のトークンを選んだ結果を示します。この方式を **greedy decoding** といい、結果は常に同じになります。

> The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French

このように同じフレーズが繰り返されやすいのも greedy decoding の特徴です。

---

ページ：**1** | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | [8](08_residual.md) | [9](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
