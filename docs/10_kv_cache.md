ページ：[00](00_quickstart.md) | [01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | **10** | [11](11_architecture.md)

---

# KV キャッシュ: 自己回帰生成の高速化

GPT-2 は自己回帰で1トークンずつ生成します（👉[09](09_output.md)）。素朴な実装では毎回全トークンを再計算しますが、KV キャッシュを使うと新しいトークンの計算だけで済みます。

1. テキスト
   - トークナイザー
     - [BPE](02_tokenizer.md)
     - [SentencePiece](03_spiece.md)
2. トークン ID 列
   - [Embedding](04_embedding.md)
3. ベクトル列
   - Transformer Block × 12
     - [LayerNorm](05_layer_norm.md)
     - [Attention](06_attention.md) — **KV キャッシュ** ← この章
     - [残差接続](08_residual.md)
     - [LayerNorm](05_layer_norm.md)
     - [MLP](07_mlp.md)
     - [残差接続](08_residual.md)
   - [最終 LayerNorm](08_residual.md)
   - [LM Head](09_output.md)
4. ロジット
   - [サンプリング](09_output.md)
5. 次のトークン

## 1. 毎回の再計算

自己回帰生成（👉[09](09_output.md)）では、トークンを1つ生成するたびに入力全体をモデルに通していました。

```
Step 1: [A, B, C]       → 全3トークンを計算 → 次のトークン D
Step 2: [A, B, C, D]    → 全4トークンを計算 → 次のトークン E
Step 3: [A, B, C, D, E] → 全5トークンを計算 → 次のトークン F
```

Transformer Block の中で他のトークンを参照するのは Attention だけです。MLP（👉[07](07_mlp.md)）、LayerNorm、Embedding、LM Head はすべてトークン単位の独立した処理です。さらに因果マスク（👉[06](06_attention.md)）により、各トークンは自分より後のトークンを参照できません。つまり Step 2 での A, B, C の Attention の計算結果は Step 1 と同じです。毎回同じ計算を繰り返すのは無駄です。

そこで、Attention で計算した K と V を保存しておき、次のステップではそれを再利用する手法が **KV キャッシュ** です。これを使うと、生成ループは次のように変わります。

```python
# Prefill: 全プロンプトを処理してキャッシュを構築
logits, kv_cache = model(np.array(input_ids), kv_cache=None)
next_token = int(np.argmax(logits[-1, :]))
input_ids.append(next_token)

# Incremental: 新トークンのみ処理
for _ in range(n_tokens_to_generate - 1):
    new_token_id = np.array([input_ids[-1]])
    logits, kv_cache = model(new_token_id, kv_cache=kv_cache)
    next_token = int(np.argmax(logits[-1, :]))
    input_ids.append(next_token)
```

初回（prefill）で全トークンを処理してキャッシュを作り、2回目以降は新しい 1 トークンだけをモデルに渡します。

### 補足：「KV キャッシュ」の名前について

「KV」という名前から、プログラミングの Key-Value ストア（辞書型）のように「キーで検索して値を取得する」仕組みを想像するかもしれません。しかし KV キャッシュの K, V は Attention の Q, K, V（👉[06](06_attention.md)）のうちの K（Key）と V（Value）を指しています。辞書型のような検索は行わず、計算済みの K, V の行列をそのまま保存しておき、次のステップで再計算を省くためのキャッシュです。

## 2. KV キャッシュの仕組み

Attention（👉[06](06_attention.md)）では、入力トークンのベクトルから Q, K, V を計算し、Q と K の内積でスコアを求め、V の加重平均を取ります。

```python
# 入力ベクトルから Q, K, V を一括計算
qkv = x @ self.w_qkv + self.b_qkv
q, k, v = np.split(qkv, 3, axis=-1)

# Q × K^T → Softmax → × V
scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)
probs = softmax(scores)
out = probs @ v
```

Q, K, V はいずれも入力 `x` から計算されます。最初の層では Embedding → LayerNorm を経た `x` から、2層目以降は前の層の出力（残差接続を含む）→ LayerNorm を経た `x` から計算されます。つまり、ある層の K, V を得るにはそれより前の全層の処理が必要です。KV キャッシュがなければ、過去のトークンについてこの一連の処理を毎回やり直すことになります。

因果マスクにより過去のトークンは未来を参照しないので、新しいトークンが追加されても過去のトークンの K, V は変わりません。そこで、計算済みの K と V を保存（キャッシュ）しておけば、過去のトークンを再処理せずに済みます。

- Prefill: `[A, B, C]` → Q, K, V を全て計算 → K, V をキャッシュに保存
- Step 1: `[D]` だけ計算 → K_D, V_D をキャッシュに追加  
  → Q_D × [K_A, ..., K_D]^T → Softmax → × [V_A, ..., V_D]
- Step 2: `[E]` だけ計算 → K_E, V_E をキャッシュに追加  
  → Q_E × [K_A, ..., K_E]^T → Softmax → × [V_A, ..., V_E]

新しいトークンの Q, K, V を計算し、K と V はキャッシュに追加します。Attention の計算では、Q は新しいトークン 1 つ分ですが、K と V はキャッシュ済みの全トークン分を使います。過去のトークンの Q は再利用する場面がないためキャッシュ不要で、保存するのは K と V だけで十分です。この非対称性が「KV キャッシュ」という名前の由来です。

なお、Q が新しいトークン 1 つ分しかないため、scores, probs, out も必然的にその 1 つ分だけになります。次のトークンの予測には最終トークンの出力だけを使う（👉[09](09_output.md)）ので、これで十分です。

## 3. 計算量の比較

Attention の計算量は Q の長さ × K の長さに比例します。

```
キャッシュなし（毎回全トークン）:
  Step 0: seq_len=4, Attention 計算量 ∝ 4×4 = 16
  Step 1: seq_len=5, Attention 計算量 ∝ 5×5 = 25
  Step 2: seq_len=6, Attention 計算量 ∝ 6×6 = 36
  Step 3: seq_len=7, Attention 計算量 ∝ 7×7 = 49

キャッシュあり（新トークンのみ）:
  Prefill: seq_len=4, Attention 計算量 ∝ 4×4 = 16
  Step 1: q_len=1, kv_len=5, Attention 計算量 ∝ 1×5 = 5
  Step 2: q_len=1, kv_len=6, Attention 計算量 ∝ 1×6 = 6
  Step 3: q_len=1, kv_len=7, Attention 計算量 ∝ 1×7 = 7
```

キャッシュなしでは合計 16+25+36+49 = 126 に対し、キャッシュありでは 16+5+6+7 = 34 です。シーケンスが長くなるほど差は広がります。Attention 以外の処理 (Embedding, LayerNorm, MLP, LM Head) もトークン単位の計算なので、新しいトークンだけ処理すれば十分です。

## 4. 実装

KV キャッシュの実装に必要な変更は、Attention, TransformerBlock, GPT2 の 3 箇所です。いずれも「キャッシュがあれば過去の K, V を再利用する」という同じ方針に基づいています。

### Attention: K, V の追加とマスクの調整

Attention 内で最も重要な変更です。新しいトークンから計算した K, V を、キャッシュ済みの K, V の末尾に追加します。Q は新しいトークン分だけですが、K と V はキャッシュを含む全トークン分になるため、スコア行列の形が変わります。

```python
# キャッシュがあれば結合
if kv_cache is not None:
    k = np.concatenate([kv_cache[0], k], axis=2)
    v = np.concatenate([kv_cache[1], v], axis=2)

# マスク: 新しいトークン(seq_len個)が全トークン(kv_len個)を参照
kv_len = k.shape[2]
mask = np.tril(np.ones((kv_len, kv_len)))[-seq_len:]
```

`seq_len` は今回処理するトークン数です。prefill 時はプロンプト全体（例: 4）、incremental 時は 1 になります。このコードは両方を統一的に扱います。

マスクの `[-seq_len:]` がポイントです。因果マスク（kv_len × kv_len の下三角行列）から最後の `seq_len` 行を取り出すことで、今回のトークンが過去のトークンを参照できるようにします。prefill 時（キャッシュなし）は `seq_len == kv_len` なので通常の因果マスクと同じで、incremental 時は最後の 1 行だけになります。

### TransformerBlock: キャッシュの受け渡し

Attention にキャッシュを渡し、更新されたキャッシュを返します。MLP はトークン単位の独立した処理（👉[07](07_mlp.md)）なので変更不要です。

```python
def __call__(self, x, kv_cache=None):
    attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, new_kv_cache
```

### GPT2: 位置埋め込みのオフセット

KV キャッシュ使用時は、新しいトークンに正しい位置を割り当てる必要があります。例えば 3 トークンの prefill 後に 4 番目のトークンを処理する場合、位置は 0 ではなく 3 です。キャッシュに保存されている系列長をオフセットとして使います。

```python
if kv_cache is not None:
    past_len = kv_cache[0][0].shape[2]
else:
    past_len = 0
positions = np.arange(past_len, past_len + seq_len)

x = self.wte[input_ids] + self.wpe[positions]
```

## 5. キャッシュの構造

KV キャッシュは各層の K と V のペアをリストとして保持します。

```
層数: 12
Layer 0 の K の形状: (12, 13, 64)  (n_head, seq_len, head_size)
Layer 0 の V の形状: (12, 13, 64)
全層のキャッシュサイズ: 1,837,056 bytes (1.8 MB)
```

13 トークン分（プロンプト 4 + 生成 9）のキャッシュで約 1.8 MB です。GPT-2 の最大コンテキスト長 1024 トークンでは約 150 MB になります。

### 補足：KV キャッシュのメモリ管理

KV キャッシュはシーケンス長に比例してメモリが増加するため、長い文脈を扱う際はボトルネックになります。最近のモデルでは Multi-Query Attention（MQA）や Grouped-Query Attention（GQA）といった手法で、複数のヘッド間で K, V を共有することでキャッシュサイズを削減しています。GPT-2 は各ヘッドが独立した K, V を持つ標準的な Multi-Head Attention を使っており、これらの最適化は含まれていません。

## 6. 速度比較

"Artificial Intelligence will" を入力とした 10 ステップの貪欲法による生成で比較します。

```
生成結果: 'Artificial Intelligence will be able to do things like create a new type'
キャッシュなし: 6.556 秒
キャッシュあり: 3.915 秒（1.7x 高速化）
生成結果一致: True
```

NumPy 実装では Attention 以外のオーバーヘッド（MLP の行列積など）も大きいため高速化は 1.5～2.0 倍程度ですが、GPU を使う実際の推論エンジンでは Attention がボトルネックとなるため効果は顕著です。また、シーケンスが長くなるほど再計算の無駄が大きくなるため、差はさらに広がります。

## まとめ

KV キャッシュによる高速化の仕組みは以下の通りです。

1. 初回 (prefill) で全トークンの Q, K, V を計算し、K と V をキャッシュに保存する
2. 位置埋め込みのオフセットにより、新しいトークンに正しい位置を割り当てる
3. 生成ステップごとに新しいトークンの K, V をキャッシュに追加する
4. 新しいトークンの Q とキャッシュ全体の K でスコアを計算し、V の加重平均を取る

因果マスクにより過去のトークンの K, V は変わらないという性質を利用した、シンプルかつ効果的な最適化です。

## 実験：KV キャッシュの効果

キャッシュあり/なしの生成結果の一致、速度比較、キャッシュの構造、計算量の比較を確認します。実行結果は本文中で引用しています。

**実行方法**: ([10_kv_cache.py](10_kv_cache.py))

```bash
uv run docs/10_kv_cache.py
```

---

ページ：[00](00_quickstart.md) | [01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | **10** | [11](11_architecture.md)
