ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | **6** | [7](07_mlp.md) | [8](08_residual.md) | [9](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# Attention: 文脈の理解

Attention は GPT-2 が「文脈」を理解するための最も重要なコンポーネントです。各トークンが他のトークンとの関係性を計算し、文脈を取り込んだベクトルへと自身を更新します。LayerNorm（[05](05_layer_norm.md) 参照）を適用した後のベクトルに対して、この処理が行われます。

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

## 1. Q, K, V（Query, Key, Value）

各トークンは、3つの役割を持つベクトルに変換されます。

*   **Query (Q)**: 「何を探しているか」（検索クエリ）
*   **Key (K)**: 「自分は何を持っているか」（検索インデックス）
*   **Value (V)**: 「実際の情報」（検索結果）

GPT-2 small では各トークンを768次元のベクトルで表し、それを12個のヘッドに分割します（各ヘッド64次元）。

```python
class AttentionParams:
    def __call__(self, x, n_head=None):
        n_head = n_head or self.n_head
        qkv = np.matmul(x, self.w_qkv) + self.b_qkv
        q, k, v = np.split(qkv, 3, axis=-1)
```

### 設計の動機: なぜ Q, K, V に分けるのか？

もし入力ベクトル同士で直接類似度を計算すると、「検索する側の顔」と「検索される側の顔」を区別できません。異なる投影行列を用意することで、同じ単語でも「文脈を探しに行く時の表現」と「文脈のヒントを与える時の表現」を使い分けられるようになります。

## 2. スコア計算とスケーリング

Q と K の内積を取り、類似度（スコア）を計算します。次元数 $\sqrt{d_k}$ で割ることでスケーリングします。

```python
scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
```

### 設計の動機: なぜ $\sqrt{d_k}$ でスケーリングするのか？

次元数が大きいと内積の値が非常に大きくなり、Softmax の出力が「ほぼ1箇所だけ1で他は0」という極端な分布になります。$\sqrt{d_k}$ で割ることで値を適正な範囲に保ち、学習を安定させます。

## 3. 因果マスキング（Causal Masking）

GPT-2 は「次の単語を予測する」モデルであるため、未来の単語を見てはいけません。下三角行列を使って、未来のスコアを $-10^{10}$ で上書きし、Softmax 後に影響が 0 になるようにします。

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

## 4. Multi-Head の統合

12個のヘッドが並列に異なる「注目パターン」を学習します。

```
最後のトークン '.' の注目先（上位3）:
  Head 0:  lazy(0.289), The(0.216),  fox(0.157)
  Head 1: .(0.991), The(0.004),  fox(0.001)
  Head 2: The(0.574),  jumps(0.075), .(0.075)
  Head 3: .(0.523),  dog(0.197),  the(0.094)
```

Head 0 は文中の重要な単語に分散して注目し、Head 1 は直前のトークンに集中するなど、各ヘッドが異なる関係性を捉えています。

結果を結合し、出力投影行列を掛けて元の次元に戻します。

```python
out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
return np.matmul(out, self.w_out) + self.b_out
```

## 実験：注目パターンの可視化

「The quick brown fox jumps over the lazy dog.」を入力し、因果マスキングの動作確認、注目先の分布、複数ヘッド間の注目パターンの違いを観察します。実行結果は本文中で引用しています。

**実行方法**: ([06_attention.py](06_attention.py))

```bash
uv run docs/06_attention.py
```

---

ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | **6** | [7](07_mlp.md) | [8](08_residual.md) | [9](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
