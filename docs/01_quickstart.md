ページ：**01** | [02](02_overview.md) | [03](03_tokenizer.md) | [04](04_spiece.md) | [05](05_embedding.md) | [06](06_layer_norm.md) | [07](07_attention.md) | [08](08_mlp.md) | [09](09_residual.md) | [10](10_output.md) | [11](11_kv_cache.md) | [12](12_architecture.md)

---

# まず動かしてみよう

技術的な詳細に入る前に、GPT-2 が実際にどう動くのかを体験しましょう。この章ではセットアップから文章生成まで、手を動かしながら GPT-2 の振る舞いを観察します。

## セットアップ

依存関係をインストールします。

```bash
uv sync
```

英語モデル（`openai-community/gpt2`）と日本語モデル（`rinna/japanese-gpt2-small`）をダウンロードします。

```bash
make download
```

## 英語テキストの生成

プロンプト（書き出し）を与えると、GPT-2 がその続きを生成します。

```bash
uv run my-gpt2 "Once upon a time"
```
> Once upon a time when no one could make impressions many people wished they had touched others who did.
> They stayed and drifted. Many don't even remember for years. One

GPT-2 は「次に来る単語を予測する」ことだけを学習したモデルです。質問に答えたり指示に従ったりする訓練は受けていませんが、書き出しの続きを確率的に生成できます。

## 日本語テキストの生成

英語ベースの GPT-2 はバイトレベル BPE により日本語の入力を受け付けますが、日本語として意味の通じる文章は生成できません。

```bash
uv run my-gpt2 -n 20 "吾輩は猫で"
```
> 吾輩は猫で自己のファンメイルに猫て是 ignored

日本語を生成するには、日本語で学習された `rinna/japanese-gpt2-small` モデルを `-m` オプションで指定します。

```bash
uv run my-gpt2 -n 20 -m rinna/japanese-gpt2-small "吾輩は猫で"
```
> 吾輩は猫で、子に親切にもした。 楽しげな娘を見るたびに、亡き妻が生きて

## サンプリングの制御

生成は確率的なため、実行するたびに異なる結果が得られます。いくつかのオプションで生成の振る舞いを調整できます。

### Temperature（温度）

`-t` で温度を指定します。低い値ほど確信度の高い単語が選ばれやすく、高い値ほど多様な出力になります。

```bash
uv run my-gpt2 -n 20 -t 0.5 "The meaning of life is"
uv run my-gpt2 -n 20 -t 1.5 "The meaning of life is"
```

温度 0.5 では堅実で予測しやすい文章、温度 1.5 では意外性のある（しかし時に支離滅裂な）文章が生成される傾向があります。

### Top-k / Top-p サンプリング

`-k` で確率上位 k 個のトークンのみを候補にし、`-p` で累積確率が p に達するまでのトークンを候補にします。

```bash
uv run my-gpt2 -n 30 -k 50 "Once upon a time"
uv run my-gpt2 -n 30 -p 0.9 "Once upon a time"
```

### 繰り返し実行

`-r` で同じプロンプトを複数回実行し、サンプリングのばらつきを観察できます。

```bash
uv run my-gpt2 -r 5 -t 0.8 -n 5 -m rinna/japanese-gpt2-small "日本の首都は東京です。中国の首都は"
```
> 日本の首都は東京です。中国の首都は北京、インドの首都デリー
> 日本の首都は東京です。中国の首都は、やはり日本の首都です
> 日本の首都は東京です。中国の首都は、上海(hua
> 日本の首都は東京です。中国の首都は北京と北京です。
> 日本の首都は東京です。中国の首都は北京ですが、東京

パターンを見せることで、モデルが持つ知識を引き出せることがわかります。「日本→東京」の対応を見せると「中国→北京」を高い確率で出力します。

## プロンプトの工夫

GPT-2 は「次の単語を予測する」Base モデルですが、プロンプトの形式を工夫することで様々な使い方ができます。

### 擬似的な対話

対話形式のプロンプトを与えると、アシスタントのように振る舞わせることができます。

```bash
uv run my-gpt2 -n 5 "User: Hello!
Assistant: Hello! How can I help you today?
User: What is the capital of France?
Assistant:"
```
> (snip)
> Assistant: Mind the capital of France

指示に従う訓練を受けていないため回答の質は低いですが、パターンを模倣しようとする性質は見て取れます。

### 執筆の呼び水

物語の書き出しから続きを生成させることで、アイデアの種として使えます。

```bash
uv run my-gpt2 -n 50 -t 0.8 "Once upon a time"
```
> Once upon a time they understood what the situation was, that he had to survive.
> So after being raised as a captive by his own father, he and his family lived a few years of
> isolation and isolation on a deserted island. It wasn't until he met

```bash
uv run my-gpt2 -n 50 -t 0.8 -m rinna/japanese-gpt2-small "昔々あるところに"
```
> 昔々あるところに、鳥が通り過ぎる。 いつもやってくる。 と。 変だわー、いろいろあってみじめに
> 感じるこの頃のことだ。 やっぱりどこかで見たような表情をしている。 その隣の小さな池が大沼

## ここまでのまとめ

ここまでで観察できたことを整理します。

- GPT-2 は**書き出しの続きを確率的に生成する**モデルである
- 温度やサンプリング手法で**生成の多様性を制御**できる
- 英語モデルと日本語モデルでは**トークナイザーが異なる**が、モデル構造は同一
- プロンプトのパターンを模倣する性質があり、**工夫次第で様々な用途**に使える
- 出力の品質は現代の LLM と比べると低いが、**基本的な仕組みは同じ**

次章からは、この「書き出しの続きを生成する」という処理が内部でどのように実現されているのかを、パイプラインの各ステップに沿って見ていきます。

---

ページ：**01** | [02](02_overview.md) | [03](03_tokenizer.md) | [04](04_spiece.md) | [05](05_embedding.md) | [06](06_layer_norm.md) | [07](07_attention.md) | [08](08_mlp.md) | [09](09_residual.md) | [10](10_output.md) | [11](11_kv_cache.md) | [12](12_architecture.md)
