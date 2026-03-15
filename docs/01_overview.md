# GPT-2 推論パイプラインの全体像

GPT-2 は「テキストを受け取り、次の単語（トークン）を予測する」モデルです。入力されたテキストは以下のパイプラインを通り、最終的に「次に来る単語の確率分布」として出力されます。

```
テキスト
  ↓ トークナイザー（02, 03 参照）
トークンID列
  ↓ Embedding（04 参照）
ベクトル列 (seq_len, 768)
  ↓ Transformer Block × 12
  │   ├ LayerNorm（05 参照）
  │   ├ Attention（06 参照）
  │   ├ 残差接続（08 参照）
  │   ├ LayerNorm
  │   ├ MLP（07 参照）
  │   └ 残差接続
  ↓ 最終 LayerNorm
  ↓ LM Head（09 参照）
ロジット (seq_len, 50257)
  ↓ サンプリング（09 参照）
次のトークン
```

---

## トークナイザー: テキストからトークンIDへ

モデルに入力する前に、テキストをトークンID（整数の列）に変換する必要があります。GPT-2 では2種類のトークナイザーを使い分けます。

**BPE トークナイザー**（02 参照）: 英語版 GPT-2 (`openai-community/gpt2`) で使用。バイトレベル BPE（Byte Pair Encoding）により、テキストをサブワード単位に分割します。未知語が発生せず、あらゆる言語のテキストを処理できます。

**SentencePiece トークナイザー**（03 参照）: 日本語版 GPT-2 (`rinna/japanese-gpt2-small`) で使用。ユニグラムモデルにより、日本語テキストを適切なサブワード単位に分割します。

---

## コードとの対応

`my_gpt2/model.py` の `GPT2.__call__` が、このパイプラインの中核です。

```python
def __call__(self, input_ids):
    # Step 1: Embedding — トークンIDをベクトルに変換し、位置情報を加算
    x = self.params.wte[input_ids] + self.params.wpe[np.arange(input_ids.shape[1])]

    # Step 2: Transformer Block × 12 — 文脈理解と特徴変換を繰り返す
    for block in self.blocks:
        x = block(x)

    # Step 3: 最終 LayerNorm — 出力前の正規化
    x = layer_norm(x, self.params.ln_f)

    # Step 4: LM Head — ベクトルを語彙サイズの確率分布に変換（Weight Tying）
    return np.matmul(x, self.params.wte.T)
```

わずか数行のコードですが、この中に Embedding、Attention、MLP、LayerNorm、残差接続、Weight Tying といった要素が凝縮されています。以降のドキュメントで、各ステップの仕組みを順番に解説していきます。

---

## 体験してみよう

### 実行方法
```bash
uv run docs/01_overview.py
```

### 実行結果（例）

入力テキスト `The capital of France is` を処理する様子を、ステップごとに追跡します。

**Step 1: Embedding**

```
  形状: (1, 5, 768)  (バッチ, トークン数, 埋め込み次元)
  平均: -0.0039, 標準偏差: 0.2418
```

5つのトークンがそれぞれ768次元のベクトルに変換されます。この時点ではまだ「文脈」を反映しておらず、各単語の静的な意味だけを持っています。

**Step 2: Transformer Block × 12**

```
    層    平均      標準偏差
    0    0.0471    2.8242
    1    0.1970   10.4311
    2    0.8200   41.4148
    ...
   10    1.2198   50.9479
   11    0.1078   14.2968
```

12層のブロックを通過するにつれ、標準偏差が急激に増加します。これは各層が文脈情報や特徴を書き込んでいることを示しています。最終層（11）で標準偏差が減少するのは、出力に向けた調整が行われているためです。

**Step 4: LM Head → 予測結果**

```
  1. ' the' (確率: 0.0846)
  2. ' now' (確率: 0.0479)
  3. ' a' (確率: 0.0462)
  4. ' France' (確率: 0.0324)
  5. ' Paris' (確率: 0.0322)
```

最終的に約5万語の語彙に対する確率分布が得られます。「Paris」が上位に入っており、モデルがフランスの首都についての知識を持っていることが分かります。

---

## 後続ドキュメント

| ドキュメント | テーマ |
|---|---|
| 02_tokenizer | BPE トークナイザー |
| 03_spiece | SentencePiece トークナイザー |
| 04_embedding | トークンIDからベクトルへの変換 |
| 05_layer_norm | Layer Normalization による正規化 |
| 06_attention | Attention による文脈の理解 |
| 07_mlp | MLP によるトークンの特徴変換 |
| 08_residual | 残差接続と Transformer Block の構造 |
| 09_output | LM Head、Weight Tying、サンプリング |
