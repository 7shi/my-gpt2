ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | **10** | [11](11_architecture.md)

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

## 1. 毎回の再計算という問題

自己回帰生成（👉[09](09_output.md)）では、トークンを1つ生成するたびに入力全体をモデルに通していました。

```
Step 1: [A, B, C]       → 全3トークンを計算 → 次のトークン D
Step 2: [A, B, C, D]    → 全4トークンを計算 → 次のトークン E
Step 3: [A, B, C, D, E] → 全5トークンを計算 → 次のトークン F
```

しかし因果マスク（👉[06](06_attention.md)）により、各トークンは自分より後のトークンを参照できません。つまり Step 2 で A, B, C の Attention 計算結果は Step 1 と同じです。毎回同じ計算を繰り返しているのは無駄です。

そこで、Attention で計算した Key と Value を保存しておき、次のステップではそれを再利用する手法が **KV キャッシュ** です。これを使うと、生成ループは次のように変わります。

```python
# Prefill: 全プロンプトを処理してキャッシュを構築
logits, kv_cache = model(np.array(input_ids), kv_cache=None)
next_token = int(np.argmax(logits[-1, :]))
input_ids.append(next_token)

# Incremental: 新トークンのみ処理
for _ in range(n_tokens_to_generate - 1):
    logits, kv_cache = model(np.array([input_ids[-1]]), kv_cache=kv_cache)
    next_token = int(np.argmax(logits[-1, :]))
    input_ids.append(next_token)
```

初回（prefill）で全トークンを処理してキャッシュを作り、2回目以降は新しい1トークンだけをモデルに渡します。

## 2. KV キャッシュの仕組み

Attention の計算は Q × K^T → Softmax → × V という流れです（👉[06](06_attention.md)）。因果マスクにより、過去のトークンの **Key（K）** と **Value（V）** は新しいトークンが追加されても変わりません。そこで、計算済みの K と V を保存（キャッシュ）しておき、新しいトークンの K, V だけを追加します。

```
Prefill:  [A, B, C] → Q,K,V を全て計算 → K,V をキャッシュに保存
Step 1:   [D] だけ計算 → K,V をキャッシュに追加 → Q_D × [K_A,K_B,K_C,K_D]^T
Step 2:   [E] だけ計算 → K,V をキャッシュに追加 → Q_E × [K_A,K_B,K_C,K_D,K_E]^T
```

新しいトークンの **Query（Q）** だけを計算し、キャッシュ済みの K, V 全体と Attention を取ります。

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

キャッシュなしでは合計 16+25+36+49 = 126 に対し、キャッシュありでは 16+5+6+7 = 34 です。シーケンスが長くなるほど差は広がります。Attention 以外の処理（Embedding、MLP、LayerNorm、LM Head）もトークン単位の計算なので、新しい1トークンだけ処理すれば十分です。

## 4. 実装

KV キャッシュの実装に必要な変更は、Attention、Transformer Block、GPT2 の3箇所です。いずれも「キャッシュがあれば過去の K, V を再利用する」という同じ方針に基づいています。

### Attention: K, V の結合とマスクの調整

Attention 内で最も重要な変更です。新しいトークンから計算した K, V を、キャッシュ済みの K, V の末尾に結合します。Q は新しいトークン分だけですが、K と V はキャッシュを含む全トークン分になるため、スコア行列の形が変わります。

```python
# キャッシュがあれば結合
if kv_cache is not None:
    k = np.concatenate([kv_cache[0], k], axis=2)
    v = np.concatenate([kv_cache[1], v], axis=2)

# マスク: 新しいトークン(q_len)が全トークン(kv_len)を参照できるようにする
kv_len = k.shape[2]
mask = np.tril(np.ones((kv_len, kv_len)))[-seq_len:]
```

マスクの `[-seq_len:]` がポイントです。因果マスク（kv_len × kv_len の下三角行列）から最後の行だけを取り出すことで、新しいトークンが過去のすべてのトークンを参照できるようにします。通常の因果マスクでは Q と K が同じ長さですが、KV キャッシュ使用時は Q が1トークン、K が全トークンという非対称な形になります。

### Transformer Block: キャッシュの受け渡し

Attention にキャッシュを渡し、更新されたキャッシュを返します。MLP はトークン単位の独立した処理（👉[07](07_mlp.md)）なので変更不要です。

```python
def __call__(self, x, kv_cache=None):
    attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
    x = x + attn_out
    x = x + self.mlp(self.ln_2(x))
    return x, new_kv_cache
```

### GPT2: 位置埋め込みのオフセット

KV キャッシュ使用時は、新しいトークンに正しい位置を割り当てる必要があります。例えば3トークンの prefill 後に4番目のトークンを処理する場合、位置は 0 ではなく 3 です。キャッシュに保存されている系列長をオフセットとして使います。

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
Layer 0 の K の形状: (1, 12, 13, 64)  (batch, n_head, seq_len, head_size)
Layer 0 の V の形状: (1, 12, 13, 64)
全層のキャッシュサイズ: 1,837,056 bytes (1794.0 KB)
```

13 トークン分（プロンプト 4 + 生成 9）のキャッシュで約 1.8 MB です。GPT-2 small の最大コンテキスト長 1024 トークンでは約 150 MB になります。

## 6. 速度比較

10 ステップの貪欲法による生成で比較します。

```
キャッシュなし: 9.212 秒
キャッシュあり: 6.064 秒（1.5x 高速化）
生成結果一致: True
```

NumPy 実装では Attention 以外のオーバーヘッド（MLP の行列積など）も大きいため高速化は 1.5 倍程度ですが、GPU を使う実際の推論エンジンでは Attention がボトルネックとなるため効果は顕著です。また、シーケンスが長くなるほど再計算の無駄が大きくなるため、差はさらに広がります。

## 実験：KV キャッシュの効果

キャッシュあり/なしの生成結果の一致、速度比較、キャッシュの構造、計算量の比較を確認します。実行結果は本文中で引用しています。

**実行方法**: ([10_kv_cache.py](10_kv_cache.py))

```bash
uv run docs/10_kv_cache.py
```

---

ページ：[01](01_overview.md) | [02](02_tokenizer.md) | [03](03_spiece.md) | [04](04_embedding.md) | [05](05_layer_norm.md) | [06](06_attention.md) | [07](07_mlp.md) | [08](08_residual.md) | [09](09_output.md) | **10** | [11](11_architecture.md)
