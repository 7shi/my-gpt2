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
プロンプトを位置引数として指定して実行します。

```bash
make run
```
または、詳細なオプションを指定して実行します。
```bash
# 英語（openai-community/gpt2、デフォルト）
uv run my-gpt2 "Once upon a time"

# 日本語（rinna/japanese-gpt2-small）
uv run my-gpt2 -m rinna/japanese-gpt2-small "吾輩は猫である"
```

#### 主なオプション:
- `prompt` (位置引数): 生成を開始するテキスト。
- `-n`, `--n_tokens`: 生成するトークン数（デフォルト: 30）。
- `-t`, `--temperature`: サンプリング温度。値を大きくすると多様性が増し、0 に近づけると決定的な生成になります（デフォルト: 1.0）。
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
# 英語
uv run my-gpt2 -n 5 "User: Hello!
Assistant: Hello! How can I help you today?
User: What is the capital of France?
Assistant:"

# 日本語
# rinna の語彙には改行が含まれないため、改行の代わりに * を使用する。
# * はモデルが区切り記号として学習しており、生成結果にも自然に現れる。
# 全角の！？も語彙外のため半角を使用する。
uv run my-gpt2 -n 10 -m rinna/japanese-gpt2-small "ユーザー: こんにちは!*アシスタント: こんにちは! 何かお手伝いできますか?*ユーザー: 日本の首都はどこですか?*アシスタント:"
```

### 2. パターンによる知識抽出（Analogy）
「AはBである。ならばCはDである」という推論能力（アナロジー）を利用します。文章を途中で止めることで、モデルが持つ知識を自然な形で引き出すことができます。

```bash
# 英語
uv run my-gpt2 -n 5 "The capital of Japan is Tokyo. The capital of France is"

# 日本語
uv run my-gpt2 -n 5 -m rinna/japanese-gpt2-small "日本の首都は東京です。フランスの首都は"
```

### 3. GPT-2 への日本語入力
英語ベースの GPT-2 は Byte-level BPE により日本語を受け付けますが、日本語として意味の通じる文章は生成できません。日本語の断片や記号が混在した出力になります。

```bash
uv run my-gpt2 -n 20 "吾輩は猫である"
# → 吾輩は猫であることは何有能はならな...‰…Gya,
```

日本語で意味の通じる文章を生成するには `rinna/japanese-gpt2-small` を使用してください。

```bash
uv run my-gpt2 -n 20 -m rinna/japanese-gpt2-small "吾輩は猫である"
# → 吾輩は猫である! というキャッチコピーに反応し、アマゾンで回答をしたため、ゾンビに見せかけた酒を
```

### 4. 執筆の呼び水・ダミーテキスト生成
物語や記事の書き出しを渡し、温度（Temperature）を調整することで、多様なアイデアを生成させることができます。

```bash
# 英語
uv run my-gpt2 -n 50 -t 0.8 "Once upon a time"

# 日本語
uv run my-gpt2 -n 50 -t 0.8 -m rinna/japanese-gpt2-small "昔々あるところに"
```

---
LLM の内部で何が起きているのかを、コードを通じて学ぶためのリポジトリです。
