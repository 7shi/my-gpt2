try:
    from importlib.metadata import version
    __version__ = version("my-gpt2")
except Exception:
    __version__ = "unknown"
