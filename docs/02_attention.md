# Multi-Head Attention と 因果マスキング 解説

Attentionは、GPT-2が「文脈」を理解するための最も重要なコンポーネントです。
各トークンが、自分自身と他のトークンの関係性を計算します。

---

### 1. Q, K, V (Query, Key, Value)

各トークンは、3つの役割を持つベクトルに変換されます。
*   **Query (Q)**: 「何を探しているか」（検索クエリ）
*   **Key (K)**: 「自分は何を持っているか」（検索インデックス）
*   **Value (V)**: 「実際の情報」（検索結果）

```python
def mha(x, w_qkv, b_qkv, w_out, b_out, n_head):
    # ... 
    # 入力を Q, K, V に一括で投影
    qkv = np.matmul(x, w_qkv) + b_qkv
    # ヘッド数に合わせて分割 (Multi-Head)
    q, k, v = np.split(qkv, 3, axis=-1)
```

---

### 2. スコア計算とスケーリング

$Q$ と $K$ の内積（Dot Product）を取り、類似度（スコア）を計算します。
次元数 $\sqrt{d_k}$ で割ることで、Softmaxが極端な値にならないよう安定させます。

```python
# q @ k.T / sqrt(d_k)
d_k = q.shape[-1]
scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
```

---

### 3. 因果マスキング (Causal Masking)

GPT-2は「次の単語を予測する」モデルであるため、未来の単語を見てはいけません。
下三角行列（Lower Triangular Matrix）を使用して、未来のスコアを非常に小さな値（$-10^{10}$）で上書きします。これにより、Softmax適用後に未来の影響が 0 になります。

```python
# 下三角行列のマスク作成
mask = np.tril(np.ones((seq_len, seq_len)))

if mask is not None:
    # 0 (未来) の部分を -inf 近似値で埋める
    scores = np.where(mask == 0, -1e10, scores)

# 正規化して重み(Probability)にする
probs = softmax(scores)
```

---

### 4. Multi-Head の統合

複数のヘッド（視点）で計算された結果を最後に結合（Concatenate）し、出力投影行列（`w_out`）を掛けて元の次元に戻します。

```python
# ヘッドを結合して元の次元に戻す
out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
# 最終的な出力投影
return np.matmul(out, w_out) + b_out
```

---

### 設計の動機：なぜ Q, K, V に分けるのか？

Attentionの設計は「検索システム」を模倣しています。
もし $x$ 同士で直接計算すると、「検索される側の顔」と「検索する側の顔」を区別できません。
あえて $W_q, W_k, W_v$ という異なる投影行列を用意することで、同じ単語であっても**「文脈を探しに行く時の顔」と「文脈のヒントを与える時の顔」を学習によって使い分ける**ことができるようになります。

### 設計の動機：なぜ $\sqrt{d_k}$ でスケーリングするのか？

ベクトルの次元数 $d_k$ が大きくなると、内積の値は非常に大きくなりやすいです。内積が大きすぎると、Softmax関数の出力が「ほぼ1箇所だけが1で他は0」という極端な状態（尖った分布）になります。
こうなると、学習時に「どの単語を重視すべきか」という情報（勾配）が消失してしまうため、$\sqrt{d_k}$ で割ることで数値を適正な範囲に保ち、学習を安定させています。
