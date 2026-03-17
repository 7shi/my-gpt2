ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | **06** | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# Attention: 文脈の理解

Attention は GPT-2 が「文脈」を理解するための最も重要なコンポーネントです。各トークンが他のトークンとの関係性を計算し、文脈を取り込んだベクトルへと自身を更新します。LayerNorm（👉[05](05_layer_norm.md)）を適用した後のベクトルに対して、この処理が行われます。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - [SentencePiece](03_spiece.md)
2. トークン ID 列
   - [Embedding](04_embedding.md)
3. ベクトル列
   - Transformer Block × 12
     - [LayerNorm](05_layer_norm.md)
     - **Attention** ← この章
     - [残差接続](08_residual.md)
     - [LayerNorm](05_layer_norm.md)
     - [MLP](07_mlp.md)
     - [残差接続](08_residual.md)
   - [最終 LayerNorm](08_residual.md)
   - [LM Head](09_output.md)
4. ロジット
   - [サンプリング](09_output.md)
5. 次のトークン

## 1. Attention

Transformer の原論文のタイトルは "Attention Is All You Need"（Vaswani et al., 2017）であり、Attention こそが Transformer の中核です。従来のモデルで使われていた再帰構造や畳み込みを排し、Attention だけで系列全体の依存関係を捉えられることを示しました。

Attention の処理は以下の流れで進みます。

1. 各トークンのベクトルから Q, K, V を生成する
2. Q と K の内積でスコア（注目度）を計算する
3. 未来のトークンをマスクで遮断する
4. Softmax でスコアを確率分布に変換する
5. 確率に応じて V を重み付き平均する

これを 12 個のヘッドで並列に行い、結果を結合して出力します。

## 2. Q, K, V（Query, Key, Value）

各トークンは、3 つの役割を持つベクトルに変換されます。

- **Q (Query)**: 「何を探しているか」（検索クエリ）
- **K (Key)**: 「自分は何を持っているか」（検索インデックス）
- **V (Value)**: 「実際の情報」（検索結果）

GPT-2 では各トークンを768次元のベクトルで表し、それを12個のヘッドに分割します（各ヘッド64次元）。

Q, K, V の生成と出力射影に使う重みとバイアスは safetensors ファイルから読み込まれます。

```python
AttentionParams(
    n_head=12,
    w_qkv=weights["h.0.attn.c_attn.weight"],  # (768, 2304) Q,K,V 結合
    b_qkv=weights["h.0.attn.c_attn.bias"],    # (2304,)
    w_out=weights["h.0.attn.c_proj.weight"],  # (768, 768) 出力射影
    b_out=weights["h.0.attn.c_proj.bias"],    # (768,)
)
```

`c_attn` の重み行列は 768×2304 で、2304 = 768×3（Q, K, V の3つ分）です。1回の行列積で Q, K, V をまとめて計算し、3分割します。

```python
class AttentionParams:
    def __call__(self, x, n_head=None):
        n_head = n_head or self.n_head
        # 768次元の入力に (768, 2304) の重み行列を掛け、Q,K,V を一括計算
        qkv = x @ self.w_qkv + self.b_qkv
        # 2304次元を3等分して Q, K, V（各768次元）に分離
        q, k, v = np.split(qkv, 3, axis=-1)
```

### 設計の動機：なぜ Q, K, V に分けるのか？

もし入力ベクトル同士で直接類似度を計算すると、「検索する側の顔」と「検索される側の顔」を区別できません。異なる投影行列を用意することで、同じ単語でも「文脈を探しに行く時の表現」と「文脈のヒントを与える時の表現」を使い分けられるようになります。

## 3. スコア計算とスケーリング

Q と K の内積を取り、類似度（スコア）を計算します。次元数 $\sqrt{d_k}$ で割ることでスケーリングします。

```python
scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(d_k)
```

### 設計の動機：なぜ $\sqrt{d_k}$ でスケーリングするのか？

次元数が大きいと内積の値が非常に大きくなり、Softmax の出力が「ほぼ1箇所だけ1で他は0」という極端な分布になります。$\sqrt{d_k}$ で割ることで値を適正な範囲に保ち、学習を安定させます。

## 4. Softmax

スコアを確率分布に変換するために **Softmax** を適用します。各スコアの指数 $e^{z_i}$ を取り、合計で割ることで、全体の和が 1 になるよう正規化します。

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

分母 $\sum_j e^{z_j}$ は統計力学の分配関数に対応します。最もスコアが高い要素に高い確率が割り当てられ、スコアの差が大きいほど確率の偏りも大きくなります。

前節のスケーリングはこの Softmax の性質と密接に関係しています。スケーリングなしでスコアが大きくなりすぎると、Softmax の出力がほぼ one-hot（1箇所だけ 1 で他は 0）になり、他のトークンからの情報を取り込めなくなります。

## 5. 因果マスキング（Causal Masking）

GPT-2 は「次の単語を予測する」モデルであるため、未来の単語を見てはいけません。下三角行列を使って、未来のスコアを $-10^{10}$（実質 $-\infty$）で上書きし、Softmax 後に確率が 0 になるようにします。

```python
mask = np.tril(np.ones((seq_len, seq_len)))
scores = np.where(mask == 0, -1e10, scores)
probs = softmax(scores)
```

4トークンの場合のマスクの形：

```
        tok0  tok1  tok2  tok3
tok0  [  s    -inf  -inf  -inf ]
tok1  [  s     s    -inf  -inf ]
tok2  [  s     s     s    -inf ]
tok3  [  s     s     s     s   ]
```

最初のトークンは自分だけに100%注目し、最後のトークンはすべてのトークンを参照できます。

```
最初のトークン 'The' の注目先:
    -> The        : 1.0000
```

## 6. Value の重み付き平均

Softmax で得られた確率分布を重みとして、V の加重平均を取ります。これが Attention の核心です。

```python
out = probs @ v  # 確率に応じて V を重み付き平均
```

たとえば最後のトークンが "The" に 0.3、"dog" に 0.5、"." に 0.2 の重みを割り当てた場合、出力ベクトルはこの3トークンの V を 3:5:2 で混合したものになります。各トークンが文脈に応じた情報を取り込んだ新しいベクトルを得る、これが「文脈の理解」の実体です。

## 7. Multi-Head の統合

ここまでが1つのヘッドの処理です。GPT-2 ではこれを12個のヘッドで並列に行い、各ヘッドが異なる「注目パターン」を学習します。

```
最後のトークン '.' の注目先（上位3）:
  Head 0:  lazy(0.289), The(0.216),  fox(0.157)
  Head 1: .(0.991), The(0.004),  fox(0.001)
  Head 2: The(0.574),  jumps(0.075), .(0.075)
  Head 3: .(0.523),  dog(0.197),  the(0.094)
```

Head 0 は文中の重要な単語に分散して注目し、Head 1 は直前のトークンに集中するなど、各ヘッドが異なる関係性を捉えています。

各ヘッドの結果（64次元 × 12ヘッド）を結合して元の768次元に戻し、出力投影行列を掛けます。

```python
out = concat_heads(out)      # (seq_len, 12×64) → (seq_len, 768)
return out @ self.w_out + self.b_out
```

## 実験：注目パターンの可視化

「The quick brown fox jumps over the lazy dog.」を入力し、因果マスキングの動作確認、注目先の分布、複数ヘッド間の注目パターンの違いを観察します。実行結果は本文中で引用しています。

**実行方法**: ([06_attention.py](06_attention.py))

```bash
uv run docs/06_attention.py
```

---

ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | **06** | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
