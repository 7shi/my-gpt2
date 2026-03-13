# Transformer Block と 残差接続 (Residual Connection) 解説

GPT-2のモデルは、この「Transformer Block」を複数（GPT-2 smallでは12個）積み重ねることで構成されています。

---

## 1. Pre-LayerNorm 構造

GPT-2の特徴的な設計の一つに、**LayerNorm (LN)** を各処理（AttentionやMLP）の**前**に置く「Pre-LayerNorm」があります。

```python
class TransformerBlock:
    def __call__(self, x):
        # Attention + 残差接続 (Pre-LayerNorm)
        x = x + mha(layer_norm(x, **self.params["ln_1"]), **self.params["attn"], n_head=self.n_head)
        
        # MLP + 残差接続 (Pre-LayerNorm)
        x = x + mlp(layer_norm(x, **self.params["ln_2"]), **self.params["mlp"])
        return x
```

### 設計の動機：なぜ Pre-LayerNorm (GPT-2) なのか？

オリジナルの Transformer (2017) では処理の「後」に LayerNorm を置いていました (Post-LN)。しかし、Post-LN は層が深くなると学習が非常に難しくなる問題がありました。
GPT-2が採用した Pre-LN では、**「残差接続のパスが遮られずにモデル全体を貫通する」**ため、信号が深層まで伝わりやすくなり、初期の学習が劇的に安定します。

---

## 2. 残差接続 (Residual Connection / Skip Connection)

`x = x + ...` という形が残差接続です。これは、層を通過する際に**「元の情報を保持したまま、変更分だけを加える」**という考え方です。
これには2つのメリットがあります：
1.  **勾配消失の防止**: 深いモデルでも誤差が直接入力側に伝わりやすくなる。
2.  **情報の伝播**: 学習の初期段階では入力をそのまま通すことができ、学習が進むにつれて必要な情報を付加していく。

---

## 3. MLP (Multi-Layer Perceptron)

MLPは、トークンごとに独立して計算を行います。Attentionが「トークン間の繋が」を扱うのに対し、MLPは「トークン単体の意味の深掘り」を担当します。

```python
def mlp(x, w_fc, b_fc, w_proj, b_proj):
    # a: 768次元から3072次元へ (x @ w_fc + b_fc)
    a = gelu(np.matmul(x, w_fc) + b_fc)
    # 3072次元から768次元に戻す
    return np.matmul(a, w_proj) + b_proj
```

### 設計の動機：なぜ MLP で次元を 4倍にするのか？

これは「カーネル法」に似た考え方です。
低次元空間では線形分離が難しい複雑なパターンも、一度高次元（4倍）に引き上げることで、特徴を整理しやすくなります。
また、**GELUという「窓」を通して情報をフィルタリング**し、再び元の次元に圧縮することで、情報の取捨選択と高度な特徴抽出を行っています。

## まとめ：Block の役割
各 Block は「文脈理解（Attention）」と「単語の深掘り（MLP）」をセットで行い、それを何度も繰り返すことで、入力された文字列の高度な意味表現を獲得していきます。
