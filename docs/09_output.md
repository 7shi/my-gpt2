ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | **09** | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# 出力: ロジットの生成とサンプリング

12層の Transformer Block と最終 LayerNorm を通過したベクトルは、最終的に「次の単語の確率分布」に変換されます。このドキュメントでは、出力層（LM Head）とサンプリング手法を解説します。

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
     - [MLP](07_mlp.md)
     - [残差接続](08_residual.md)
   - [最終 LayerNorm](08_residual.md)
   - **LM Head** ← この章
4. ロジット
   - **サンプリング** ← この章
5. 次のトークン

## 1. LM Head (Weight Tying)

通常、出力層には独立した重み行列を用意しますが、GPT-2 では入力の Embedding 行列（WTE）をそのまま出力層にも再利用します。この手法を **Weight Tying** と呼びます。これは次のようなメリットがあります。

1. **セマンティックな一貫性**: 「単語 A を表すベクトル」と「次の単語として A を予測するベクトル」は同じ意味空間にあるべき
2. **パラメータ効率**: GPT-2 の 124M パラメーターのうち約 30% を占める行列を共有することで、メモリを節約

WTE の形状は (50257, 768) で、行が語彙、列が埋め込み次元です。最終層の出力 x の形状は (len(x), 768) なので、そのまま掛けることはできません。WTE を転置して (768, 50257) にすることで、行列の積が成立します。

```python
# LM Head: WTE の転置行列を掛ける (Weight Tying)
return x @ self.wte.T  # → (len(x), 50257)
```

行列積 `@` によって、各トークンのベクトルと全語彙の埋め込みベクトルとの内積を一括計算しています。出力は確率分布の前段階となるスコアで、**ロジット** (logit) と呼ばれます。

内積はベクトルが同じ方向を向いているほど値が大きくなるという性質を持つため、ロジットは「モデルの最終出力に最も近い埋め込みベクトルを持つトークンほど大きな値を持つ」という特徴があります。

## 2. ロジットから確率へ

ロジットは生のスコアであり、負の値や非常に大きな値も含むため、そのままでは確率として扱えません。Softmax（👉[06](06_attention.md)）を適用することで、すべての値が 0〜1 の範囲に収まり、合計が 1 になる確率分布に変換されます。

ロジットが最も高いトークンに最も高い確率が割り当てられ、スコアの差が大きいほど確率の偏りも大きくなります。

```python
probs = softmax(next_token_logits)  # ロジットを確率分布に変換
```

## 3. サンプリング手法

Softmax で得た確率分布から次のトークンを選ぶ方法によって、生成されるテキストの性質が変わります。

### 貪欲法

常に最も確率が高いトークンを選択します。決定論的でランダム性がないため結果は常に同じになりますが、長い文章ではループが発生しやすくなります。

```python
next_token = int(np.argmax(next_token_logits))
```

ループは局所最適で、窪地にハマり込んで抜けられなくなっているような状態です。その外側に脱出するためには、ランダム性を加える必要があります。

### Temperature

ロジットを温度 T で割ってから Softmax を適用することで、確率分布の「尖り具合」を調整します。

```python
# ロジットを温度 T で割り、分布の尖り具合を調整
next_token_logits = next_token_logits / temperature
# Softmax で確率分布に変換（ボルツマン分布と同形）
probs = softmax(next_token_logits)
# 得られた確率分布に従ってトークンをランダムに選択
next_token = int(np.random.choice(len(probs), p=probs))
```

以下は Softmax 後の確率の例です。

```
T=0.5（集中）: ' be' 0.787, ' become' 0.029, ' not' 0.029
T=1.0（通常）: ' be' 0.184, ' become' 0.036, ' not' 0.035
T=2.0（平坦）: ' be' 0.013, ' become' 0.006, ' not' 0.006
```

- **低温度（T < 1.0）**: 高確率トークンに集中し、保守的な生成
- **高温度（T > 1.0）**: 確率が平坦化し、多様で創造的な生成

ゼロで割ることはできませんが、T = 0 は前述の貪欲法として扱われます。

※ 「温度」という用語は統計力学に由来します。Softmax はボルツマン分布 $p_i = e^{-E_i / T} / Z$（$Z$ は分配関数）と同じ形をしており、T が分布の広がりを制御するパラメーターとして対応します。

### Top-k Sampling

上位 k 個のトークン以外を $-\infty$ にマスクしてから Softmax を適用します。候補数が常に k 個に固定されます。

```python
# ロジットが大きい上位 k 個のインデックスを取得
top_k_indices = np.argpartition(next_token_logits, -top_k)[-top_k:]
# 全要素を -∞ で埋めたマスクを作成
mask = np.full_like(next_token_logits, -np.inf)
# 上位 k 個だけ元のロジットを残す（残りは Softmax で確率 0 になる）
mask[top_k_indices] = next_token_logits[top_k_indices]
next_token_logits = mask
```

### Top-p Sampling（Nucleus Sampling）

Softmax 適用後の確率を高い順に並べ、累積確率が p に達するまでのトークンだけを候補にします。語彙の分布に応じて候補数が動的に変わるため、Top-k より適応的です。

```python
# 確率の高い順にインデックスをソート
sorted_indices = np.argsort(probs)[::-1]
# ソート順に累積確率を計算
cumulative_probs = np.cumsum(probs[sorted_indices])
# 累積確率が p を超える位置を求め、そこまでを候補とする
cutoff = np.searchsorted(cumulative_probs, top_p) + 1
# 候補外のトークンの確率を 0 にする
probs[sorted_indices[cutoff:]] = 0.0
# 確率の合計が 1 になるよう再正規化
probs /= probs.sum()
```

Top-k と Top-p は組み合わせることができ、その場合は Top-k でマスクした後に Top-p を適用します。

## 4. 自己回帰生成

GPT-2 は一度に1トークンずつ予測します。予測したトークンを入力に追加して再び推論するループ（自己回帰）を繰り返すことで文章を生成します。

```python
for _ in range(n_tokens_to_generate):
    # トークン列全体をモデルに入力し、各位置のロジットを得る
    logits = model(np.array(input_ids))
    # 最後の位置のロジットから最も確率の高いトークンを選択（貪欲法）
    next_token = int(np.argmax(logits[-1, :]))
    # 予測トークンを入力に追加して次のステップへ
    input_ids.append(next_token)
```

毎回入力全体をモデルに通し、最後のトークンの確率分布から次のトークンを選びます。この例は貪欲法（最も確率が高いトークンを選択）ですが、Temperature や Top-k/Top-p を組み合わせたサンプリングも可能です。

```
Step 1: 'Artificial Intelligence will'
  → 候補: ' be'(0.184), ' become'(0.036)...
  → 選択: ' be'
Step 2: 'Artificial Intelligence will be'
  → 候補: ' able'(0.095), ' a'(0.085)...
  → 選択: ' able'
Step 3: 'Artificial Intelligence will be able'
  → 候補: ' to'(0.991)
  → 選択: ' to'（確信度が非常に高い）
```

Step 3 では「be able」の後に「to」が来る確率が99%を超えています。文法的なパターンが確定すると、モデルの「迷い」は消え、決定論的な振る舞いになります。

なお、この素朴な実装では毎回全トークンを再計算しています。次回に説明する KV キャッシュを使えば、新しいトークンの計算だけで済むようになります。

## 実験：Temperature と自己回帰生成

Temperature による確率分布の変化、3ステップ分の自己回帰生成の過程、Weight Tying の詳細を確認します。実行結果は本文中で引用しています。

**実行方法**: ([09_output.py](09_output.py))

```bash
uv run docs/09_output.py
```

---

ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | **09** | [10](10_kv_cache.md) | [11](11_architecture.md)
