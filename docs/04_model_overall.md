# GPT-2 モデル全体の構成 と LM Head (Weight Tying) 解説

Transformer Blockを積み上げた後、モデルがどのようにして最終的な「次の単語」の確率を算出しているかを解説します。

---

### 1. 入力層：Embedding

入力されたトークンID（整数）をベクトルに変換します。GPT-2では、単語の意味と位置の2つのベクトルを足し合わせます。

```python
def __call__(self, input_ids):
    # wte: Word Token Embedding (語彙のベクトル化)
    # wpe: Word Position Embedding (位置情報の付加)
    x = self.params["wte"][input_ids] + self.params["wpe"][np.arange(input_ids.shape[1])]
    
    # 多数の Transformer Block を通過
    for block in self.blocks:
        x = block(x)
```

---

### 2. 出力層：LM Head と Weight Tying

モデルの最終出力を、再び「語彙サイズ（約5万）」の確率分布に変換します。
「IDからベクトルにする行列 (`wte`)」と「ベクトルからIDに戻す行列 (`LM Head`)」を、**同じもの（の転置）** として扱います。

```python
# 最終的な LayerNorm を適用
x = layer_norm(x, **self.params["ln_f"])

# LM Head: WTE の転置行列を掛ける (Weight Tying)
return np.matmul(x, self.params["wte"].T)
```

---

### 設計の動機：なぜ位置ベクトル (WPE) を「足す」のか？

Attention機構は、入力がどんな順番で並んでいても同じ結果を計算してしまいます（順序不変性）。
順番を「追加の入力（特徴量）」として足し合わせることで、モデルに「これは1番目の単語だ」という情報を埋め込んでいます。

### 設計の動機：なぜ重み共有 (Weight Tying) をするのか？

1. **セマンティックな一貫性**: 
   「単語 $A$ を表現するベクトル」と「次の単語として $A$ を予測するベクトル」は、本質的に同じ意味空間にあるべきです。同じ重みを使うことで、単語の類似性が予測にも直接反映されます。
2. **圧倒的な節約**: 
   GPT-2 (small) のパラメータ数 124M のうち、Embedding 行列だけで約 38M (約30%) を占めます。これを共有することで、モデルの精度を保ったままメモリ使用量を劇的に削減できます。
