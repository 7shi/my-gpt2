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

### 3. 自己回帰的な推論 (Generation) とサンプリング

推論時には、この出力の最後尾（最新の単語）の確率分布から次のトークンを選択し、それを再び入力に追加して推論を繰り返します（自己回帰）。

#### サンプリング手法：Greedy vs Temperature
トークンを選択する手法には主に以下の 2 つがあります：

1. **Greedy Search (Temperature = 0)**:
   常に最も確率が高いトークンを 1 つだけ選びます。
   - **特徴**: 決定的で一貫性がありますが、文章が長くなると同じフレーズを繰り返す「ループ現象」が発生しやすくなります。

2. **Temperature Sampling (Temperature > 0)**:
   ソフトマックスを適用する前のロジットを温度 $T$ で割り、分布を変形させてからランダムにサンプリングします。
   - **低温度 (T < 1.0)**: 確率の高いトークンに集中し、保守的で安全な生成になります。
   - **高温度 (T > 1.0)**: 確率の低いトークンにもチャンスを与え、多様で創造的な（時には支離滅裂な）生成になります。

```python
# 実装の抜粋 (my_gpt2/generate.py)
if temperature > 0:
    # 温度でロジットをスケールし、確率分布からランダム選択
    next_token_logits = next_token_logits / temperature
    probs = softmax(next_token_logits)
    next_token = int(np.random.choice(len(probs), p=probs))
else:
    # 最も確率が高いものを常に選択
    next_token = int(np.argmax(next_token_logits))
```

### まとめ：推論の全プロセス
1.  テキストをトークナイザーで数値(ID)化。
2.  Embeddingでベクトル化し、位置情報を足す。
3.  複数の Transformer Block を通して文脈を深める。
4.  Weight Tying を使って語彙確率に変換。
5.  サンプリング手法（温度）を用いて次の単語を決定し、再び入力へ。

---

### 設計の動機：なぜ位置ベクトル (WPE) を「足す」のか？

Attention機構は、入力がどんな順番で並んでいても同じ結果を計算してしまいます（順序不変性）。
順番を「追加の入力（特徴量）」として足し合わせることで、モデルに「これは1番目の単語だ」という情報を埋め込んでいます。

### 設計の動機：なぜ重み共有 (Weight Tying) をするのか？

1. **セマンティックな一貫性**: 
   「単語 $A$ を表現するベクトル」と「次の単語として $A$ を予測するベクトル」は、本質的に同じ意味空間にあるべきです。同じ重みを使うことで、単語の類似性が予測にも直接反映されます。
2. **圧倒的な節約**: 
   GPT-2 (small) のパラメータ数 124M のうち、Embedding 行列だけで約 38M (約30%) を占めます。これを共有することで、モデルの精度を保ったままメモリ使用量を劇的に削減できます。
