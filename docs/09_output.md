ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | [8](08_residual.md) | **9** | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# 出力: ロジットの生成とサンプリング

12層の Transformer Block と最終 LayerNorm を通過したベクトルは、最終的に「次の単語の確率分布」に変換されます。このドキュメントでは、出力層（LM Head）とサンプリング手法を解説します。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - [SentencePiece](03_spiece.md)
2. トークンID列
   - [Embedding](04_embedding.md)
3. ベクトル列 (seq_len, 768)
   - Transformer Block × 12
     - [LayerNorm](05_layer_norm.md)
     - [Attention](06_attention.md)
     - [残差接続](08_residual.md)
     - [LayerNorm](05_layer_norm.md)
     - [MLP](07_mlp.md)
     - [残差接続](08_residual.md)
   - [LayerNorm](08_residual.md)
   - **LM Head** ← この章
4. ロジット (seq_len, 50257)
   - **サンプリング** ← この章
5. 次のトークン

## 1. LM Head（Weight Tying）

最終 LayerNorm（08 参照）を通過したベクトルは、**Weight Tying** によって語彙サイズの確率分布に変換されます。

```python
# LM Head: WTE の転置行列を掛ける (Weight Tying)
return np.matmul(x, self.wte.T)
```

出力は **ロジット（logit）** と呼ばれる生のスコアで、語彙の各トークンに対する「次に来る可能性」を表します。

### Weight Tying: なぜ入力と出力で同じ行列を共有するのか？

WTE（Embedding 行列）は入力でトークンIDをベクトルに変換し、その転置は出力でベクトルをトークンIDに戻します。

```
WTE 形状: (50257, 768)    入力: トークンID → ベクトル
WTE.T:    (768, 50257)    出力: ベクトル → ロジット
共有パラメータ数: 38,597,376 (147.2 MB)
```

1. **セマンティックな一貫性**: 「単語Aを表すベクトル」と「次の単語としてAを予測するベクトル」は同じ意味空間にあるべき
2. **パラメータ効率**: GPT-2 small の 124M パラメータのうち約30%を占める行列を共有し、メモリを節約

## 2. サンプリング手法

ロジットから次のトークンを選ぶ方法によって、生成されるテキストの性質が変わります。

### Temperature

ロジットを温度 $T$ で割ることで確率分布の「尖り具合」を調整します。

```
T=0.5（集中）: ' be' 0.787, ' become' 0.029, ' not' 0.029
T=1.0（通常）: ' be' 0.184, ' become' 0.036, ' not' 0.035
T=2.0（平坦）: ' be' 0.013, ' become' 0.006, ' not' 0.006
```

- **低温度（T < 1.0）**: 高確率トークンに集中し、保守的な生成
- **高温度（T > 1.0）**: 確率が平坦化し、多様で創造的な生成

### Greedy Search（Temperature = 0）

常に最も確率が高いトークンを選択します。決定的ですが、長い文章ではループが発生しやすくなります。

### Top-k Sampling

上位 $k$ 個のトークン以外を $-\infty$ にマスクしてから Softmax を適用します。候補数が常に $k$ 個に固定されます。

### Top-p Sampling（Nucleus Sampling）

Softmax 後の確率を高い順に並べ、累積確率が $p$ に達するまでのトークンだけを候補にします。語彙の分布に応じて候補数が動的に変わるため、Top-k より適応的です。

Top-k と Top-p は組み合わせることができ、その場合は Top-k でマスクした後に Top-p を適用します。

## 3. 自己回帰生成

GPT-2 は一度に1トークンずつ予測します。予測したトークンを入力に追加して再び推論するループ（自己回帰）を繰り返すことで文章を生成します。

```python
for _ in range(n_tokens_to_generate):
    logits = model(np.array(input_ids))
    next_token = int(np.argmax(logits[-1, :]))
    input_ids.append(next_token)
```

毎回入力全体をモデルに通し、最後のトークンの確率分布から次のトークンを選びます。この例は greedy（最も確率が高いトークンを選択）ですが、Temperature や Top-k/Top-p を組み合わせたサンプリングも可能です。

```
Step 1: 'Artificial Intelligence will' → 候補: ' be'(0.184), ' become'(0.036)...
  → 選択: ' be'
Step 2: 'Artificial Intelligence will be' → 候補: ' able'(0.095), ' a'(0.085)...
  → 選択: ' able'
Step 3: 'Artificial Intelligence will be able' → 候補: ' to'(0.991)
  → 選択: ' to'（確信度が非常に高い）
```

Step 3 では「be able」の後に「to」が来る確率が99%を超えています。文法的なパターンが確定すると、モデルの「迷い」は消え、決定的な振る舞いになります。

なお、この素朴な実装では毎回全トークンを再計算しています。KV キャッシュ（10 参照）を使えば、新しいトークンの計算だけで済むようになります。

## コマンドライン引数

```
-t / --temperature  サンプリング温度（デフォルト: 1.0、0で Greedy）
-k / --top_k        top-k の k（デフォルト: なし）
-p / --top_p        top-p の p（デフォルト: なし、0〜1の実数）
```

## 実験：Temperature と自己回帰生成

Temperature による確率分布の変化、3ステップ分の自己回帰生成の過程、Weight Tying の詳細を確認します。実行結果は本文中で引用しています。

**実行方法**: ([09_output.py](09_output.py))

```bash
uv run docs/09_output.py
```

---

ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | [8](08_residual.md) | **9** | [10](10_kv_cache.md) | [11](11_architecture.md)
