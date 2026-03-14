# GPT-2 モデル全体の構成 と LM Head (Weight Tying) 解説

Transformer Blockを積み上げた後、モデルがどのようにして最終的な「次の単語」の確率を算出しているかを解説します。

---

## 1. 入力層：Embedding

入力されたトークンID（整数）をベクトルに変換します。GPT-2では、単語の意味と位置の2つのベクトルを足し合わせます。

GPT-2 small の Embedding 行列の形は次のとおりです。

```
WTE: (50257, 768)   # 語彙数 × embed_dim  （Word Token Embedding）
WPE: (1024,  768)   # 最大シーケンス長 × embed_dim  （Word Position Embedding）
```

WTE でトークンIDを768次元のベクトルに変換し、WPE で位置ベクトルを加算します。

```python
def __call__(self, input_ids):
    # wte: Word Token Embedding (語彙のベクトル化)
    # wpe: Word Position Embedding (位置情報の付加)
    x = self.params["wte"][input_ids] + self.params["wpe"][np.arange(input_ids.shape[1])]

    # 多数の Transformer Block を通過
    for block in self.blocks:
        x = block(x)
```

### 設計の動機：なぜ位置ベクトル (WPE) を「足す」のか？

Attention機構は、入力がどんな順番で並んでいても同じ結果を計算してしまいます（順序不変性）。
順番を「追加の入力（特徴量）」として足し合わせることで、モデルに「これは1番目の単語だ」という情報を埋め込んでいます。

---

## 2. 出力層：LM Head と Weight Tying

モデルの最終出力を、再び「語彙サイズ（約5万）」の確率分布に変換します。
「IDからベクトルにする行列 (`wte`)」と「ベクトルからIDに戻す行列 (`LM Head`)」を、**同じもの（の転置）** として扱います。

```python
# 最終的な LayerNorm を適用
x = layer_norm(x, **self.params["ln_f"])

# LM Head: WTE の転置行列を掛ける (Weight Tying)
return np.matmul(x, self.params["wte"].T)
```

### 設計の動機：なぜ重み共有 (Weight Tying) をするのか？

1. **セマンティックな一貫性**:
   「単語 $A$ を表現するベクトル」と「次の単語として $A$ を予測するベクトル」は、本質的に同じ意味空間にあるべきです。同じ重みを使うことで、単語の類似性が予測にも直接反映されます。
2. **圧倒的な節約**:
   GPT-2 (small) のパラメータ数 124M のうち、Embedding 行列だけで約 38M (約30%) を占めます。これを共有することで、モデルの精度を保ったままメモリ使用量を劇的に削減できます。

---

## 3. 自己回帰的な推論 (Generation) とサンプリング

推論時には、この出力の最後尾（最新の単語）の確率分布から次のトークンを選択し、それを再び入力に追加して推論を繰り返します（自己回帰）。

### サンプリング手法

**ロジット（logit）**とは Softmax を適用する前の生のスコアのことです。Temperature サンプリングでは、ロジットを温度 $T$ で割ることで確率分布の「尖り具合」を調整します。

ロジット例 `[2.0, 1.0, 0.5]` に対してTemperatureを変えた場合の変化を示します。

```
T=1.0 （通常）: ロジット [2.0, 1.0, 0.5]  → softmax → [0.59, 0.24, 0.17]
T=0.5 （集中）: ÷0.5 → [4.0, 2.0, 1.0]  → softmax → [0.84, 0.11, 0.05]
T=2.0 （平坦）: ÷2.0 → [1.0, 0.5, 0.25] → softmax → [0.46, 0.34, 0.20]
```

トークンを選択する手法には主に以下の 4 つがあります：

1. **Greedy Search (Temperature = 0)**:
   常に最も確率が高いトークンを 1 つだけ選びます。
   - **特徴**: 決定的で一貫性がありますが、文章が長くなると同じフレーズを繰り返す「ループ現象」が発生しやすくなります。

2. **Temperature Sampling (Temperature > 0)**:
   ソフトマックスを適用する前のロジットを温度 $T$ で割り、分布を変形させてからランダムにサンプリングします。
   - **低温度 (T < 1.0)**: 確率の高いトークンに集中し、保守的で安全な生成になります。
   - **高温度 (T > 1.0)**: 確率の低いトークンにもチャンスを与え、多様で創造的な（時には支離滅裂な）生成になります。

3. **Top-k Sampling**:
   Temperature 適用後のロジットのうち、上位 $k$ 個のトークン以外を $-\infty$ にマスクしてから Softmax をかけます。確率が極端に低いトークンを候補から除外することで、生成の質を安定させます。
   - **特徴**: 候補数が常に $k$ 個に固定されるため、語彙の分布によっては適切でないことがあります。

4. **Top-p Sampling（Nucleus Sampling）**:
   Softmax 後の確率を高い順に並べ、累積確率が $p$ に達するまでのトークンだけを候補にします。語彙の分布に応じて候補数が動的に変わるため、top-k より適応的です。
   - **例**: $p=0.9$ のとき、上位トークンが確率0.6・0.25・0.10なら累積0.95で3トークンが候補。上位1トークンだけで0.95を超える場合はそれ1つのみになります。

top-k と top-p は組み合わせることができ、その場合は top-k でマスクした後に top-p をさらに適用します。

```python
# 実装の抜粋 (my_gpt2/generate.py)
for _ in range(n_tokens_to_generate):
    logits = model(inputs)
    next_token_logits = logits[0, -1, :]

    # サンプリング
    if temperature > 0:
        next_token_logits = next_token_logits / temperature

        # top-k: 上位k個以外を -inf にマスク
        if top_k is not None and top_k > 0:
            top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
            mask = np.full_like(next_token_logits, -np.inf)
            mask[top_k_indices] = next_token_logits[top_k_indices]
            next_token_logits = mask

        probs = softmax(next_token_logits)

        # top-p: 累積確率が p に達するまでのトークンのみ残す
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumulative_probs = np.cumsum(probs[sorted_indices])
            cutoff = np.searchsorted(cumulative_probs, top_p) + 1
            removed = sorted_indices[cutoff:]
            probs[removed] = 0.0
            probs /= probs.sum()

        next_token = int(np.random.choice(len(probs), p=probs))
    else:
        next_token = int(np.argmax(next_token_logits))
```

コマンドライン引数との対応：

```
-t / --temperature  サンプリング温度（デフォルト: 1.0、0で Greedy）
-k / --top_k        top-k の k（デフォルト: なし）
-p / --top_p        top-p の p（デフォルト: なし、0〜1の実数）
```

## まとめ：推論の全プロセス
1.  テキストをトークナイザーで数値(ID)化。
2.  Embeddingでベクトル化し、位置情報を足す。
3.  複数の Transformer Block を通して文脈を深める。
4.  Weight Tying を使って語彙確率に変換。
5.  サンプリング手法（Temperature / Top-k / Top-p）を用いて次の単語を決定し、再び入力へ。
