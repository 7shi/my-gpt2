from pathlib import Path

try:
    from importlib.metadata import version
    __version__ = version("my-gpt2")
except Exception:
    __version__ = "unknown"

# 重みの置き場所（model_id のサブディレクトリを含む親ディレクトリ）。
# 各関数・クラスで weights_dir が省略されたときに参照される。
# 既定はこのパッケージの1つ上（リポジトリ直下）の weights。
# 編集可能インストール（pip install -e . / uv tool install -e .）ならリポジトリを指す。
WEIGHTS_DIR = str(Path(__file__).resolve().parent.parent / "weights")
