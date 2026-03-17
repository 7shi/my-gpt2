ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | **07** | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# MLP: トークンの特徴変換

MLP（Multi-Layer Perceptron）は、各トークンを独立に処理する特徴変換です。Attention が「トークン間の関係」を扱うのに対し、MLP は「トークン単体の意味の深掘り」を担当します。LayerNorm を適用した後のベクトルに対して、この処理が行われます。

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
     - **MLP** ← この章
     - [残差接続](08_residual.md)
   - [最終 LayerNorm](08_residual.md)
   - [LM Head](09_output.md)
4. ロジット
   - [サンプリング](09_output.md)
5. 次のトークン

## 1. 次元の拡張と圧縮

GPT-2 small の MLP は、768次元のベクトルを一度4倍の3072次元に拡張し、GELU 活性化を適用してから、元の768次元に戻します。

```
入力:   (seq_len, 768)
中間層: (seq_len, 3072)   ← 4倍に拡張 + GELU
出力:   (seq_len, 768)    ← 元に戻す
```

```python
class MLPParams:
    def __call__(self, x):
        a = gelu(x @ self.w_fc + self.b_fc)    # 768 → 3072
        return a @ self.w_proj + self.b_proj   # 3072 → 768
```

### 設計の動機: なぜ4倍に拡張するのか？

低次元空間では線形分離が難しい複雑なパターンも、高次元に引き上げることで分離しやすくなります（カーネル法に似た考え方）。GELU で情報をフィルタリングしてから元の次元に圧縮することで、情報の取捨選択と高度な特徴抽出を行っています。

## 2. GELU 活性化関数

**GELU（Gaussian Error Linear Unit）** は ReLU の滑らか版です。ReLU が $x < 0$ で完全にゼロにするのに対し、GELU は確率的に抑制します。

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

実際の中間層での GELU の効果を見てみます。

```
中間層の要素数: 15360
負の入力: 13520 (88.0%)
GELU で抑制（|出力| < 0.01）: 656 (4.3%)
活性化後にほぼゼロ（|出力| < 0.1）: 7419 (48.3%)
```

3072次元の約半数が GELU によってほぼゼロに抑制されています。これは「情報のフィルタリング」として機能し、各トークンにとって重要な特徴だけを選択的に通過させています。

## 3. MLP によるベクトルの変化

MLP 適用前後のベクトルを比較すると、大きな変化が起きていることが分かります。

```
      トークン   入力std   出力std   コサイン類似度
       The    0.2058    4.4876      0.3055
   capital    0.1636    1.0964      0.3250
        of    0.1539    1.0827      0.2095
    France    0.1494    1.2973      0.3210
        is    0.1472    1.1864      0.1841
```

コサイン類似度が0.2〜0.3と低く、MLP がベクトルの方向を大きく変えていることが分かります。標準偏差も大幅に増加しており、MLP が情報を積極的に書き込んでいます。

## 実験：次元変化と活性化の効果

MLP の次元変化（768→3072→768）、GELU による活性化の効果、入出力ベクトルの変化量を確認します。実行結果は本文中で引用しています。

**実行方法**: ([07_mlp.py](07_mlp.py))

```bash
uv run docs/07_mlp.py
```

---

ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | **07** | [08](08_residual.md) | [09](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
