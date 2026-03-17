ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | **8** | [9](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)

---

# 残差接続と Transformer Block

これまでの章で、GPT-2 を構成する個々の部品を見てきました。ここではそれらがどう組み合わさるかを確認します。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - [SentencePiece](03_spiece.md)
2. トークン ID 列
   - [Embedding](04_embedding.md)
3. ベクトル列
   - **Transformer Block × 12** ← この章
     - [LayerNorm](05_layer_norm.md)
     - [Attention](06_attention.md)
     - **残差接続** ← この章
     - [LayerNorm](05_layer_norm.md)
     - [MLP](07_mlp.md)
     - **残差接続** ← この章
   - **最終 LayerNorm** ← この章
   - [LM Head](09_output.md)
4. ロジット
   - [サンプリング](09_output.md)
5. 次のトークン

## 1. 残差接続 (Residual Connection)

Transformer Block の内部は次の構造です。GPT-2 small ではこのブロックを12個積み重ねることで、テキストの高度な意味表現を獲得します。

```
Input ─────┐
  ↓        │
LayerNorm  │
  ↓        │
Attention  │
  ↓        │
(+) ←──────┘ 残差接続
  ├────────┐
  ↓        │
LayerNorm  │
  ↓        │
MLP        │
  ↓        │
(+) ←──────┘ 残差接続
  ↓
出力 x
```

`(+)` が**残差接続**で、各処理の出力を元の入力に加算する仕組みです。

コードで見ると、`x = x + f(x)` という形が残差接続です。層の出力をそのまま使うのではなく、「元の入力に変更分を加える」という考え方です。

```python
class TransformerBlock:
    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

残差接続には2つのメリットがあります。

1. **勾配消失の防止**: 深いモデルでも誤差が入力側に直接伝わりやすくなる
2. **情報の伝播**: 学習初期は入力をそのまま通し、学習が進むにつれて必要な情報を付加していく

Attention も MLP も、基本演算は行列積と非線形関数の組み合わせであり、1958 年に提案されたパーセプトロン（`y = activation(W·x + b)`）の延長線上にあります。しかし素朴にパーセプトロンを何十層も積むと勾配消失で学習できません。残差接続が「バイパス道路」を作ることで深い層の積み重ねを可能にし、Transformer のスケーリングを支えています。

## 2. ブロック内部の処理ステップ

1つのブロック内で、ベクトルがどのように変化するかを見てみます。ここでは入力テキスト "Machine Learning" を例に使います。GPT-2 の Attention は過去方向のみ参照するため（👉[06](06_attention.md)）、最後のトークンはそれより前のすべてのトークンを「見ている」唯一の位置です。そのため文全体の情報が集約されやすく、以降の観察では最後のトークン "Learning" の 768 次元ベクトルに注目します。標準偏差は、この 768 個の成分のばらつきを表します。

| ステップ | 標準偏差 | 備考 |
|---|---|---|
| 入力 | 0.2230 | |
| LayerNorm | 0.1127 | |
| Attention | 1.2763 | |
| 残差接続 | 1.3099 | 残差1 = 入力 + Attention |
| LayerNorm | 0.1560 | |
| MLP | 1.0851 | |
| 残差接続 | 2.0801 | 残差1 + MLP |

LayerNorm で値を正規化した後、Attention や MLP が情報を書き込み、残差接続で元のベクトルに加算されます。標準偏差が増加しているのは、各処理が新しい情報を積み重ねていることを示しています。

## 3. 12 ブロックの積み重ね

12 層のブロックを通過するにつれ、ベクトルは Embedding 時の表現から大きく変化します。引き続き最後のトークンで観察します。

| 層 | 標準偏差 | Embedding からのコサイン類似度 |
|---|---|---|
| Embedding | 0.2230 | 1.0000 |
| 0 | 2.0801 | 0.2038 |
| 1 | 2.1621 | 0.1738 |
| 2 | 2.3903 | 0.1430 |
| ... | | |
| 10 | 8.7932 | 0.0025 |
| 11 | 14.2092 | -0.0441 |
| ln_f | 7.9525 | -0.0033 |

初期の Embedding ベクトルとのコサイン類似度が急速に低下し、層10ではほぼ0になります。これは、ブロックを重ねるごとに元の「静的な単語の意味」から「文脈を反映した豊かな表現」へと変化していくことを意味しています。

### 最終 LayerNorm (ln_f)

12 層のブロックを通過した後、出力層 (LM Head) に渡す前に最終 LayerNorm が適用されます。

```python
x = self.ln_f(x)
```

表の最終行を見ると、標準偏差が 14.2 → 7.95 に縮小しています。これは出力層に渡す前にスケールを整える役割です。コサイン類似度は -0.04 → -0.003 とわずかに変化しています。

ここで注意が必要なのは、LayerNorm は通常の「ベクトルの正規化」（長さを1にする）とは異なるという点です。通常の正規化はベクトルの方向を変えないためコサイン類似度に影響しませんが、LayerNorm は次のような操作を行います。

1. 平均を引く
2. 標準偏差で割る
3. 学習済みの gamma/beta で各成分を個別に変換する

特に 1. の平均の引き算と 3. の成分ごとのアフィン変換はベクトルの方向を変えるため、コサイン類似度にも影響を与えます。この効果はセマンティック検索のセクションで具体的に確認します。

## 4. 文脈付き埋め込み

同じ単語「bank」でも、周囲の文脈によって異なるベクトルに変化します。

- 文 A: "The river bank was covered"
- 文 B: "The money bank was covered"

"bank" に相当するトークンのベクトルが、文 A と文 B とでどのように変化するかを比較してみます。

| 層 | コサイン類似度 | |
|---|--:|---|
| Embedding | 1.0000 | 同じ単語なので完全一致 |
| 0 | 0.9421 | |
| 1 | 0.9315 | |
| ... | | |
| 5 | 0.8258 | 最も乖離 |
| ... | | |
| 11 | 0.9326 | |
| ln_f | 0.9830 | |

Embedding 層では同一のベクトルですが、Attention によって周囲の単語（river / money）の情報を取り込むことで、層を経るごとに異なるベクトルへと分岐していきます。これが **文脈付き埋め込み（Contextual Embedding）** です。

ln_f を通すと類似度が 0.9326 → 0.9830 に上昇し、文脈による違いが縮小しています。トークン全体で見た標準偏差の縮小と同様に、LayerNorm がベクトル間の方向の差を均一化する効果が現れています。

## RAG やセマンティック検索との関連

現代の文章検索（セマンティック検索）や RAG で使われる「文章ベクトル」は、まさにこの Transformer の出力です。セクション2で述べたように、最後のトークンのベクトルは文章全体の情報が集約されているため、文章の意味を要約した表現として利用できます。

### 文章間の類似度

GPT-2 の最後のトークンのベクトルで文章間のコサイン類似度を計算してみます。「The cat sat on the mat.」を基準に、他の9文と比較します。

```
基準文: 'The cat sat on the mat.'（LayerNorm なし）
 1.  0.9892  A kitten was resting on the rug.
 2.  0.9643  It is raining heavily outside.
 3.  0.9531  The stock market crashed yesterday.
 4.  0.9443  He likes to read novels at night.
 5.  0.9442  Investors lost money in the financial crisis.
 6.  0.9436  She enjoys reading books before bed.
 7.  0.9400  The weather is sunny and warm today.
 8.  0.9355  Dogs are loyal and friendly animals.
 9.  0.9304  Python is a popular programming language.
```

kitten の文が1位、programming の文が最下位と、傾向自体は正しいのですが、全体が 0.93〜0.99 の狭い範囲に集中しています。

最終 LayerNorm（ln_f）を通すと、類似度が 0.995〜0.999 にさらに圧縮され、順位にも影響が出ます。

```
基準文: 'The cat sat on the mat.'（LayerNorm あり）
 1.  0.9990  A kitten was resting on the rug.
 2.  0.9981  Dogs are loyal and friendly animals.
 3.  0.9981  It is raining heavily outside.
 4.  0.9978  The weather is sunny and warm today.
 5.  0.9973  She enjoys reading books before bed.
 6.  0.9972  He likes to read novels at night.
 7.  0.9960  Investors lost money in the financial crisis.
 8.  0.9959  The stock market crashed yesterday.
 9.  0.9958  Python is a popular programming language.
```

1位と最下位は同じですが、中間の順位は入れ替わっています。例えば Dogs（動物）が8位→2位に上昇しており、cat との意味的な関連性がより反映されているようにも見えます。

類似度の絶対的なレンジは 0.003 と狭いですが、比較は相対的なものなので、スケールの基準が変わっただけとも言えます。LayerNorm によって隠れ状態の正規化が行われることで、意味的な関係がより表面化している可能性があります。

### キーワード検索

キーワード1語で文章を検索してみます。

```
キーワード: 'programming'（LayerNorm なし）
 1.  0.9092  The cat sat on the mat.
 2.  0.8969  A kitten was resting on the rug.
 3.  0.8862  The stock market crashed yesterday.
 4.  0.8841  It is raining heavily outside.
 5.  0.8735  Investors lost money in the financial crisis.
 6.  0.8697  Python is a popular programming language.
 7.  0.8569  Dogs are loyal and friendly animals.
 8.  0.8560  The weather is sunny and warm today.
 9.  0.8483  He likes to read novels at night.
10.  0.8478  She enjoys reading books before bed.
```

期待する文が 6 位と、まったく機能していません。単語 1 つと文章ではトークン系列の長さが異なるため、隠れ状態のスケールに差が生じ、コサイン類似度が歪みます。

LayerNorm を通してスケールを正規化すると改善します。

```
キーワード: 'programming'（LayerNorm あり）
 1.  0.9908  Python is a popular programming language.
 2.  0.9858  The weather is sunny and warm today.
 3.  0.9854  A kitten was resting on the rug.
 4.  0.9850  She enjoys reading books before bed.
 5.  0.9850  Dogs are loyal and friendly animals.
 6.  0.9839  The cat sat on the mat.
 7.  0.9832  It is raining heavily outside.
 8.  0.9811  He likes to read novels at night.
 9.  0.9776  The stock market crashed yesterday.
10.  0.9767  Investors lost money in the financial crisis.
```

正しい文が 1 位になりました。文章間では順位の差を圧縮してしまう LayerNorm ですが、キーワード検索では系列長の違いによるスケール差を正規化する効果があります。ただし "animal" や "finance" では改善せず、意味的な検索としてはまだ不十分です。

### GPT-2 と専用埋め込みモデル

GPT-2 は「次の単語の予測」に特化して学習されたモデルであり、文章の意味的な類似度を区別するようには最適化されていません。

RAG などで使われる専用の埋め込みモデル（OpenAI の `text-embedding-3-small` や BERT 系モデルなど）は、「意味が似た文章をより近づけ、異なる文章をより遠ざける」ための追加の学習（**対照学習**）が行われているため、文章間の類似度がよりはっきりと分かれます。

実は、これらの埋め込みモデルも Embedding → Transformer Blocks → LayerNorm という GPT-2 と同じ構造を持っています。BERT 系は Attention が双方向（全トークンが全トークンを見る）である点が異なりますが、`intfloat/e5-mistral-7b-instruct` のように GPT 系デコーダ（因果的 Attention）をそのまま骨格に使う埋め込みモデルも存在します。つまり、同じ構造のモデルでも、何を目的に学習したかによって出力ベクトルの性質が変わるのです。

## 実験：ブロック内外の表現変化

1ブロック内の各処理ステップの標準偏差、12ブロック全体を通したベクトルの変化、文脈による同一単語（bank）の表現の違い、文章レベルの埋め込みによる類似度を確認します。実行結果は本文中で引用しています。

**実行方法**: ([08_residual.py](08_residual.py))

```bash
uv run docs/08_residual.py
```

---

ページ：[1](01_overview.md) | [2](02_tokenizer.md) | [3](03_spiece.md) | [4](04_embedding.md) | [5](05_layer_norm.md) | [6](06_attention.md) | [7](07_mlp.md) | **8** | [9](09_output.md) | [10](10_kv_cache.md) | [11](11_architecture.md)
