ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | **8** | [9](09_output.md)

---

# 残差接続と Transformer Block

Transformer Block は、Attention（06 参照）と MLP（07 参照）を組み合わせ、**残差接続**で結合した処理単位です。GPT-2 small ではこのブロックを12個積み重ねることで、テキストの高度な意味表現を獲得します。

## 1. 残差接続（Residual Connection）

`x = x + f(x)` という形が残差接続です。層の出力をそのまま使うのではなく、「元の入力に変更分を加える」という考え方です。

```python
class TransformerBlock:
    def __call__(self, x):
        x = x + mha(layer_norm(x, self.params.ln_1), self.params.attn, n_head=self.n_head)
        x = x + mlp(layer_norm(x, self.params.ln_2), self.params.mlp)
        return x
```

2つのメリットがあります。
1.  **勾配消失の防止**: 深いモデルでも誤差が入力側に直接伝わりやすくなる
2.  **情報の伝播**: 学習初期は入力をそのまま通し、学習が進むにつれて必要な情報を付加していく

## 2. ブロック内部の処理ステップ

1つのブロック内で、ベクトルがどのように変化するかを見てみます。GPT-2 の Attention は過去方向のみ参照するため（06 参照）、最後のトークンはそれより前のすべてのトークンを「見ている」唯一の位置です。そのため文全体の情報が集約されやすく、以降の観察では最後のトークン「Learning」の768次元ベクトルに注目します。標準偏差は、この768個の成分のばらつきを表します。

```
入力              標準偏差: 0.2230
LayerNorm 1 後    標準偏差: 0.1127
Attention 出力    標準偏差: 1.2763
残差接続 1 後     標準偏差: 1.3099  (入力 + Attention)
LayerNorm 2 後    標準偏差: 0.1560
MLP 出力          標準偏差: 1.0851
残差接続 2 後     標準偏差: 2.0801  (残差1 + MLP)
```

LayerNorm で値を正規化した後、Attention や MLP が情報を書き込み、残差接続で元のベクトルに加算されます。標準偏差が増加しているのは、各処理が新しい情報を積み重ねていることを示しています。

## 3. 12ブロックの積み重ね

12層のブロックを通過するにつれ、ベクトルは Embedding 時の表現から大きく変化します。引き続き最後のトークンで観察します。

```
  層    標準偏差   Emb からの cos 類似度
  Emb    0.2230       1.0000
    0    2.0801       0.2038
    1    2.1621       0.1738
    2    2.3903       0.1430
    ...
   10    8.7932       0.0025
   11   14.2092      -0.0441
```

初期の Embedding ベクトルとのコサイン類似度が急速に低下し、層10ではほぼ0になります。これは、ブロックを重ねるごとに元の「静的な単語の意味」から「文脈を反映した豊かな表現」へと変化していくことを意味しています。

## 4. 文脈付き埋め込み

同じ単語「bank」でも、周囲の文脈によって異なるベクトルに変化します。

```
文A: 'The river bank was covered'
文B: 'The money bank was covered'

  層    文A-文B cos 類似度
  Emb        1.0000   ← 同じ単語なので完全一致
    0        0.9421
    1        0.9315
    ...
    5        0.8258   ← 最も乖離
    ...
   11        0.9326
```

Embedding 層では同一のベクトルですが、Attention によって周囲の単語（river / money）の情報を取り込むことで、層を経るごとに異なるベクトルへと分岐していきます。これが **文脈付き埋め込み（Contextual Embedding）** です。

### RAG やセマンティック検索との関連

現代の文章検索（セマンティック検索）や RAG で使われる「文章ベクトル」は、まさにこの Transformer の出力です。セクション2で述べたように、最後のトークンのベクトルは文章全体の情報が集約されているため、文章の意味を要約した表現として利用できます。

実際に GPT-2 の最後のトークンのベクトルで文章間の類似度を計算してみます。「The cat sat on the mat.」を基準に、他の9文との cos 類似度を比較します。

```
基準文: 'The cat sat on the mat.'
cos 類似度  文
    0.9892  A kitten was resting on the rug.       ← 最も類似
    0.9643  It is raining heavily outside.
    ...
    0.9355  Dogs are loyal and friendly animals.
    0.9304  Python is a popular programming language.  ← 最も非類似
```

kitten/rug の文が1位、programming の文が最下位と、傾向自体は正しいのですが、全体が 0.93〜0.99 の狭い範囲に集中しています。

キーワード1語での検索も試してみます。

```
キーワード: 'programming'（LayerNorm なし）
cos 類似度  文
    0.9092  The cat sat on the mat.                    ← 無関係な文が1位
    ...
    0.8697  Python is a popular programming language.

キーワード: 'programming'（LayerNorm あり）
cos 類似度  文
    0.9908  Python is a popular programming language.  ← 正しい文が1位
    ...
    0.9767  Investors lost money in the financial crisis.
```

LayerNorm なしでは無関係な文が上位に来てしまいます。単語1つと文章ではトークン系列の長さが異なるため、隠れ状態のスケールに差が生じ、cos 類似度が歪むのです。LayerNorm を通すとスケールが正規化されるため、キーワード "programming" では正しい文が1位になります。ただし "animal" や "finance" では改善せず、意味的な検索としてはまだ不十分です。

GPT-2 は「次の単語の予測」に特化して学習されたモデルであり、文章の意味的な類似度を区別するようには最適化されていません。

RAG などで使われる専用の埋め込みモデル（OpenAI の `text-embedding-3-small` や BERT 系モデルなど）は、「意味が似た文章をより近づけ、異なる文章をより遠ざける」ための追加の学習（**対照学習**）が行われているため、文章間の類似度がよりはっきりと分かれます。

## 実験：ブロック内外の表現変化

1ブロック内の各処理ステップの標準偏差、12ブロック全体を通したベクトルの変化、文脈による同一単語（bank）の表現の違い、文章レベルの埋め込みによる類似度を確認します。実行結果は本文中で引用しています。

**実行方法**: ([08_residual.py](08_residual.py))

```bash
uv run docs/08_residual.py
```

---

ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | **8** | [9](09_output.md)
