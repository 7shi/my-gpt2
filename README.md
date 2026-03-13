# my-gpt2: GPT-2 Scratch Implementation with NumPy

このプロジェクトは、LLM（大規模言語モデル）のアーキテクチャを深く理解するために、GPT-2（124Mモデル）を **NumPyのみ** を使用してスクラッチで実装したものです。PyTorchやTensorFlowなどの深層学習フレームワークを使用せず、行列演算レベルからTransformerの仕組みを再現しています。

**約400行のPythonコード**で、トークナイザー、モデル本体、重みロード、文章生成までの全工程を完結させているのが特徴です。

## 🚀 主な特徴

- **フレームワーク不使用**: 行列演算ライブラリ `NumPy` のみを用いた GPT-2 推論エンジン。
- **自作 BPE トークナイザー**: OpenAI の公式仕様（バイトレベル BPE）に準拠したトークナイザーを独自に実装。
- **公式重みのロード**: Hugging Face で公開されている `safetensors` 形式の学習済み重みを読み込み、推論（文章生成）が可能。
- **テスト駆動開発 (TDD)**: 各コンポーネント（Attention, MLP, LayerNorm 等）が数学的に正しいことを `pytest` で検証済み。
- **詳細な解説ドキュメント**: 「なぜその設計になっているのか」という動機（Motivation）を含めた技術解説を完備。

## 📚 技術解説

詳細な解説は `docs/` ディレクトリにあります。各コンポーネントの数学的な意味や設計の動機をまとめています。

1. [01_tokenizer.md](docs/01_tokenizer.md): バイトレベル BPE と Unicode マッピング
2. [02_attention.md](docs/02_attention.md): Multi-Head Attention と 因果マスキング
3. [03_block.md](docs/03_block.md): Pre-LayerNorm と 残差接続
4. [04_model_overall.md](docs/04_model_overall.md): 全体構成と 重み共有 (Weight Tying)

## 📁 ディレクトリ構成

```text
my-gpt2/
├── my_gpt2/
│   ├── model.py      # GPT-2 アーキテクチャ本体 (Transformer)
│   ├── tokenizer.py  # 自作 BPE トークナイザー
│   ├── loader.py     # 重みロードとマッピング
│   └── generate.py   # 文章生成実行スクリプト
├── docs/             # 技術解説ドキュメント (01〜04)
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
公式の GPT-2 重み (`model.safetensors`) とトークナイザー用ファイル (`vocab.json`, `merges.txt`) をダウンロードします。

```bash
make download
```

### 3. 文章生成の実行
プロンプトを位置引数として指定して実行します。

```bash
make run
```
または、詳細なオプションを指定して実行します。
```bash
# 基本的な実行
uv run python my_gpt2/generate.py Once upon a time

# 日本語プロンプト、トークン数、温度の指定
uv run python my_gpt2/generate.py こんにちは、私の名前は -n 50 -t 0.7
```

#### 主なオプション:
- `prompt` (位置引数): 生成を開始するテキスト。
- `-n`, `--n_tokens`: 生成するトークン数（デフォルト: 30）。
- `-m`, `--model`: 使用するモデル ID またはパス（デフォルト: `openai-community/gpt2`）。
- `-t`, `--temperature`: サンプリング温度。値を大きくすると多様性が増し、0 に近づけると決定的な生成になります（デフォルト: 1.0）。

## 🧪 テスト
各モジュールの正当性を確認するためにテストを実行できます。

```bash
uv run pytest
```

---
LLM の内部で何が起きているのかを、コードを通じて学ぶためのリポジトリです。
