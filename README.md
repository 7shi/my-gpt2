# my-gpt2: GPT-2 Scratch Implementation with NumPy

このプロジェクトは、LLM（大規模言語モデル）のアーキテクチャを深く理解するために、[GPT-2](https://huggingface.co/openai-community/gpt2)（124Mモデル）の推論エンジンをスクラッチで実装したものです。PyTorchやTensorFlowなどの深層学習フレームワークを使用せず、NumPyによってTransformerの仕組みを再現しています。rinna社が開発した日本語モデル [japanese-gpt2-small](https://huggingface.co/rinna/japanese-gpt2-small) にも対応しています。

初期実装（GPT-2 による英語テキスト生成が動作するまで）は、Gemini CLI を使い、Gemini 3 Flash Preview モデルとの対話を通じて**約400行・約2時間**で構築されました。

その後、Claude Code を使いながら、日本語モデル対応や SentencePiece トークナイザーの実装など、追加開発を継続しています。

## 🚀 主な特徴

- **フレームワーク不使用**: 行列演算ライブラリ `NumPy` のみを用いた GPT-2 推論エンジン。
- **自作 BPE トークナイザー**: OpenAI の公式仕様（バイトレベル BPE）に準拠したトークナイザーを独自に実装。
- **自作 SentencePiece トークナイザー**: `rinna/japanese-gpt2-small` 向けのユニグラムモデルを外部ライブラリなしで実装。`-m` オプションでモデルを切り替えるだけで日本語生成に対応。
- **公式重みのロード**: Hugging Face で公開されている `safetensors` 形式の学習済み重みを読み込み、推論（文章生成）が可能。
- **テスト駆動開発 (TDD)**: 各コンポーネント（Attention, MLP, LayerNorm 等）が数学的に正しいことを `pytest` で検証済み。
- **詳細な解説ドキュメント**: 「なぜその設計になっているのか」という動機（Motivation）を含めた技術解説を完備。

## 📚 技術解説

詳細な解説は `docs/` ディレクトリにあります。各コンポーネントの数学的な意味や設計の動機をまとめています。

1. [01_tokenizer.md](docs/01_tokenizer.md): バイトレベル BPE と Unicode マッピング
2. [02_attention.md](docs/02_attention.md): Multi-Head Attention と 因果マスキング
3. [03_block.md](docs/03_block.md): Pre-LayerNorm と 残差接続
4. [04_model_overall.md](docs/04_model_overall.md): 全体構成と 重み共有 (Weight Tying)
5. [05_spiece.md](docs/05_spiece.md): SentencePiece トークナイザー（ユニグラムモデル）

## 🔍 GPT-2 の位置づけ

GPT-2（124M パラメータ）の生成結果を見ると、文法的な整合性はあるものの意味の一貫性は低く、実用には程遠い印象を受けます。しかしそれは問題ではなく、むしろ GPT-2 はその「途中段階」を記録した歴史的なモデルです。

現代の LLM（GPT-4、Claude など）も、Transformer アーキテクチャの基本は 2017 年の "Attention is All You Need" からほぼ変わっていません。GPT-2 との本質的な違いはパラメータ数と学習データの規模です。OpenAI は「スケールすれば意味が通る振る舞いが自然に生まれる（創発する）」という仮説を持ち、それに賭けました。GPT-3（175B、GPT-2 の約1400倍）でその仮説が劇的に実証され、few-shot learning が突然機能し始めました。

つまり GPT-2 は、現代の LLM への道筋を理解するうえで最もコンパクトな出発点です。このリポジトリは「GPT-2 で実用的なものを作る」のではなく、**現代の LLM の内部で何が起きているかをコードで理解する**ことを目的としています。

## 📁 ディレクトリ構成

```text
my-gpt2/
├── my_gpt2/
│   ├── model.py      # GPT-2 アーキテクチャ本体 (Transformer)
│   ├── tokenizer.py  # 自作 BPE トークナイザー
│   ├── spiece.py     # 自作 SentencePiece トークナイザー（rinna 向け）
│   ├── loader.py     # 重みロードとマッピング
│   └── generate.py   # 文章生成実行スクリプト
├── weights/
│   ├── openai-community/
│   │   └── gpt2/                  # make download-gpt2 で生成
│   └── rinna/
│       └── japanese-gpt2-small/   # make download-rinna で生成
├── docs/             # 技術解説ドキュメント (01〜05)
├── tests/            # ユニットテスト
├── Makefile          # セットアップと実行の自動化
└── pyproject.toml    # プロジェクト設定 (hatchling)
```

## 🛠️ セットアップと使用方法

### 1. 環境構築
`uv` を使用して依存関係をインストールします。

```bash
uv sync
```

### 2. 重みと語彙ファイルのダウンロード
公式の GPT-2 重み (`model.safetensors`) とトークナイザー用ファイルをダウンロードします。

```bash
make download          # 両モデルをまとめてダウンロード
make download-gpt2     # openai-community/gpt2 のみ
make download-rinna    # rinna/japanese-gpt2-small のみ
```

### 3. 文章生成の実行
プロンプトを引数として指定して実行します。デフォルトは GPT-2 モデルです。

```bash
uv run my-gpt2 "Once upon a time"
```
> Once upon a time when no one could make impressions many people wished they had touched others who did.
> They stayed and drifted. Many don't even remember for years. One

英語ベースの GPT-2 は Byte-level BPE により日本語を受け付けますが、日本語として意味の通じる文章は生成できません。日本語の断片や記号が混在した出力になります。

```bash
uv run my-gpt2 -n 20 "吾輩は猫で"
```
> 吾輩は猫で自己のファンメイルに猫て是 ignored

日本語で意味の通じる文章を生成するには `rinna/japanese-gpt2-small` を使用してください。

```bash
uv run my-gpt2 -n 20 -m rinna/japanese-gpt2-small "吾輩は猫で"
```
> 吾輩は猫で、子に親切にもした。 楽しげな娘を見るたびに、亡き妻が生きて

#### 主なオプション:
- `prompt` (位置引数): 生成を開始するテキスト。
- `-n`, `--n_tokens`: 生成するトークン数（デフォルト: 30）。
- `-t`, `--temperature`: サンプリング温度。値を大きくすると多様性が増し、0 に近づけると決定的な生成になります（デフォルト: 1.0）。
- `-k`, `--top_k`: Top-k サンプリング。確率上位 k 個のトークンのみを候補にします（デフォルト: なし）。
- `-p`, `--top_p`: Top-p サンプリング（Nucleus Sampling）。累積確率が p に達するまでのトークンを候補にします（デフォルト: なし）。
- `-m`, `--model`: モデルID（デフォルト: `openai-community/gpt2`、例: `rinna/japanese-gpt2-small`）。

## 🧪 テスト
各モジュールの正当性を確認するためにテストを実行できます。

```bash
uv run pytest
```

## 💡 Baseモデルの実用的な活用法

本プロジェクトで実装した GPT-2 は、現代のチャット専用モデル（Instructモデル）とは異なり、純粋に「次の単語を予測する」ための **Baseモデル** です。しかし、プロンプトの工夫（Few-shot / Completion）次第で以下のような実用的な使い方が可能です。

### 1. 擬似的な対話（Chat）
指示に従う訓練（Instruct）を受けていないモデルでも、「対話の記録」というパターンを模したプロンプトを渡すことで、アシスタントのように振る舞わせることができます。

```bash
uv run my-gpt2 -n 5 "User: Hello!
Assistant: Hello! How can I help you today?
User: What is the capital of France?
Assistant:"
```
> (snip)  
> Assistant: Mind the capital of France

rinna の語彙には改行が含まれないため、ここでは改行の代わりに `*` を使用します。

```bash
uv run my-gpt2 -n 10 -m rinna/japanese-gpt2-small "ユーザー: こんにちは!*アシスタント: こんにちは! 何かお手伝いできますか?*ユーザー: 日本の首都はどこですか?*アシスタント:"
```
> （略）  
> アシスタント: 日本はどのくらい日本と呼ばれますか?*

### 2. パターンによる知識抽出（Analogy）
「AはBである。ならばCはDである」という推論能力（アナロジー）を利用します。文章を途中で止めることで、モデルが持つ知識を自然な形で引き出すことができます。

```bash
uv run my-gpt2 -n 5 "The capital of Japan is Tokyo. The capital of France is"
```
> The capital of Japan is Tokyo. The capital of France is Paris. The capital of

```bash
uv run my-gpt2 -n 5 -m rinna/japanese-gpt2-small "日本の首都は東京です。フランスの首都は"
```
> 日本の首都は東京です。フランスの首都はヴェルサイユですが、いつも

### 3. 執筆の呼び水・ダミーテキスト生成
物語や記事の書き出しを渡し、温度（Temperature）を調整することで、多様なアイデアを生成させることができます。

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

---
LLM の内部で何が起きているのかを、コードを通じて学ぶためのリポジトリです。
