ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | **5** | [6](06_attention.md) | [7](07_mlp.md) | [8](08_residual.md) | [9](09_output.md)

---

# Layer Normalization: 各層への入力の正規化

Transformer Block 内の各処理（Attention、MLP）の前に、入力ベクトルを正規化する **LayerNorm** が適用されます。GPT-2 全体で合計25回使われる基礎的な処理です。

## 1. LayerNorm の仕組み

LayerNorm は、各トークンのベクトル（768次元）を **平均0・分散1** に正規化してから、学習可能なパラメータ **γ**（スケール）と **β**（シフト）を適用します。

```python
def layer_norm(x, params, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return params.g * (x - mean) / np.sqrt(variance + eps) + params.b
```

### 正規化の効果

Embedding 直後のベクトルは、トークンごとに分散がばらついています。

```
        トークン      平均        分散
         The     -0.0068      0.1336
     capital     -0.0016      0.0508
          of     -0.0042      0.0363
      France     -0.0024      0.0399
```

正規化によって、すべてのトークンが平均0・分散1に統一されます。

### γ と β の役割

正規化だけではモデルの表現力が制限されるため、学習済みの γ と β で値を調整します。

```
γ: 平均=0.1804, 標準偏差=0.0413
β: 平均=-0.0066, 標準偏差=0.0358
```

γ の平均が約0.18であることから、正規化後のベクトルは元の約1/5のスケールに縮小されています。これにより、後続の Attention や MLP に渡す値の範囲が適切に制御されます。

## 2. Pre-LayerNorm vs Post-LayerNorm

オリジナルの Transformer（2017年）では処理の**後**に LayerNorm を置いていました（Post-LN）。GPT-2 は処理の**前**に置く **Pre-LayerNorm** を採用しています。

```python
# Pre-LayerNorm（GPT-2）
x = x + mha(layer_norm(x, ln_1), ...)    # LN → Attention → 残差接続
x = x + mlp(layer_norm(x, ln_2), ...)    # LN → MLP → 残差接続
```

### 設計の動機

Post-LN では、残差接続のパスが LayerNorm によって遮られるため、層が深くなると学習が不安定になる問題がありました。Pre-LN では残差接続のパスがモデル全体を貫通するため、信号が深層まで伝わりやすくなり、学習が安定します。

## 3. GPT-2 での使用箇所

```
各ブロック: ln_1（Attention 前）, ln_2（MLP 前）× 12ブロック = 24回
最終出力:   ln_f（LM Head 前）× 1回
合計: 25回
```

## 体験してみよう

### 実行方法
```bash
uv run docs/05_layer_norm.py
```

Embedding 直後のベクトルに対して、(1) 正規化のみ、(2) γ/β 適用後の統計を比較し、LayerNorm がどのように値を変換しているかを確認できます。

---

ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | **5** | [6](06_attention.md) | [7](07_mlp.md) | [8](08_residual.md) | [9](09_output.md)
