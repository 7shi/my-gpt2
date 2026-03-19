ページ：[00](00_quickstart.md) | [01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | **04** | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# Embedding: トークンIDからベクトルへ

GPT-2 の推論パイプラインの最初のステップは、トークンID（整数）をベクトル（数値のリスト）に変換する **Embedding** です。GPT-2 では、単語の意味と位置の2つのベクトルを足し合わせます。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - [SentencePiece](03_spiece.md)
2. トークン ID 列
   - **Embedding** ← この章
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

## 1. WTE（Word Token Embedding）: 単語の意味

トークンをベクトル（数字を並べたもの）で表現することを**埋め込み** (Embedding) と呼びます。また、このベクトルは**埋め込みベクトル**と呼ばれます。意味が似ている埋め込みベクトル同士の「向き」は近くなります。

WTE は語彙中の各トークンに対応する埋め込みベクトルを格納した巨大な行列です。GPT-2 のトークン数は 50,257 で埋め込みベクトルは 768 次元のため、float32 で格納すれば約 147.2 MBになります

トークン ID によって WTE から行を取り出せば、そのトークンに対応する埋め込みベクトルが得られます。

```python
v = wte[input_id]  # (768,)
```

トークン ID のリストを渡せば、複数の埋め込みベクトルを並べた行列が取得できます。プログラミングの用語で言えば、リスト内包表記による map 操作と同じです。

```python
x = wte[input_ids]  # (len(input_ids), 768)
# 以下と等価：np.array([wte[id] for id in input_ids])
```

### コサイン類似度

2つのベクトルの「向き」の近さを測る指標がコサイン類似度です。（1.0 で完全一致、0.0 で無関係、-1.0 で正反対）

```
cat      <-> dog      : 0.3815
cat      <-> apple    : 0.2480  （動物同士の方が類似度が高い）
man      <-> woman    : 0.5863  （非常によく似た文脈で使われる）
```

### ベクトル演算

埋め込み空間では意味の演算が可能になることがあります。`king - man + woman` を計算すると：

```
king       (類似度: 0.7752)
woman      (類似度: 0.5815)
women      (類似度: 0.4754)
```

期待した queen はトップ5に入りませんが、「女性性」の方向へベクトルが移動していることは確認できます。GPT-2 の WTE は「次の単語を予測する」ために最適化されたものであり、意味の類推に特化したモデル（Word2Vec など）とは目的が異なるためです。

## 2. WPE（Word Position Embedding）: 位置情報

GPT-2 が使用する Attention 機構は、入力がどんな順番で並んでいても同じ結果を計算してしまいます（順序不変性）。位置情報を明示的に与えなければ、"I love you" と "you love I" を区別できません。

位置を区別するためのベクトルを並べた行列が WPE です。GPT-2 で扱えるトークン数は最大 1,024 なので、1024 個の位置ベクトルがあります。

```
WPE: (1024, 768)   # 最大シーケンス長 × 埋め込み次元
```

近い位置ほど類似度が高く、遠い位置ほど低くなります。

```
位置 0 <-> 位置   1 : 0.5242
位置 0 <-> 位置  10 : 0.4443
位置 0 <-> 位置 100 : 0.2991
位置 0 <-> 位置 500 : 0.0612
```

## 3. 埋め込みの合成

WTE と WPE を**足し合わせる**ことで、「単語の意味」と「位置情報」を兼ね備えたベクトルが得られます。

```python
x = self.wte[input_ids] + self.wpe[np.arange(len(input_ids))]
```

例として "The capital of France is" を入力した場合を示します。

```
トークン: ['The', ' capital', ' of', ' France', ' is']
WTE 形状: (5, 768)    # 各トークンの意味ベクトル
WPE 形状: (5, 768)    # 各位置の位置ベクトル
合成後:   (5, 768)    # 意味 + 位置
```

### 設計の動機：なぜ連結（concatenate）ではなく加算なのか？

連結すると次元が倍になり、後続の全ての行列サイズも倍になります。加算なら次元を維持でき、パラメータ数を抑えられます。加算しても学習によって位置と意味が自然に分離されることが知られています。

## 実験：単語ベクトルの類似度と演算

WTE による単語間のコサイン類似度、最近傍探索、ベクトル演算（king − man + woman）に加え、WPE の位置間類似度と埋め込みの合成を確認します。実行結果は本文中で引用しています。

**実行方法**: ([04_embedding.py](04_embedding.py))

```bash
uv run docs/04_embedding.py
```

---

ページ：[00](00_quickstart.md) | [01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | **04** | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
