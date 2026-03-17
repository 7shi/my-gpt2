import numpy as np
from dataclasses import dataclass

_no_cache = object()

@dataclass
class LayerNormParams:
    g: np.ndarray
    b: np.ndarray

    def __call__(self, x, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        return self.g * (x - mean) / np.sqrt(variance + eps) + self.b

@dataclass
class AttentionParams:
    w_qkv: np.ndarray
    b_qkv: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray
    n_head: int = None

    def __call__(self, x, n_head=None, kv_cache=_no_cache):
        n_head = n_head or self.n_head
        seq_len, embed_dim = x.shape
        qkv = x @ self.w_qkv + self.b_qkv

        q, k, v = np.split(qkv, 3, axis=-1)
        head_size = embed_dim // n_head

        def split_heads(tensor):
            return tensor.reshape(seq_len, n_head, head_size).transpose(1, 0, 2)

        q, k, v = map(split_heads, [q, k, v])

        if kv_cache is not _no_cache:
            if kv_cache is not None:
                k = np.concatenate([kv_cache[0], k], axis=1)
                v = np.concatenate([kv_cache[1], v], axis=1)
            kv_len = k.shape[1]
            mask = np.tril(np.ones((kv_len, kv_len)))[-seq_len:]
            out = attention(q, k, v, mask=mask)
            out = out.transpose(1, 0, 2).reshape(seq_len, embed_dim)
            return out @ self.w_out + self.b_out, (k, v)

        mask = np.tril(np.ones((seq_len, seq_len)))
        out = attention(q, k, v, mask=mask)

        out = out.transpose(1, 0, 2).reshape(seq_len, embed_dim)
        return out @ self.w_out + self.b_out

@dataclass
class MLPParams:
    w_fc: np.ndarray
    b_fc: np.ndarray
    w_proj: np.ndarray
    b_proj: np.ndarray

    def __call__(self, x):
        a = gelu(x @ self.w_fc + self.b_fc)
        return a @ self.w_proj + self.b_proj

@dataclass
class BlockParams:
    ln_1: LayerNormParams
    attn: AttentionParams
    ln_2: LayerNormParams
    mlp: MLPParams

@dataclass
class GPT2Params:
    wte: np.ndarray
    wpe: np.ndarray
    ln_f: LayerNormParams
    blocks: list[BlockParams]

def gelu(x):
    """
    GELU（Gaussian Error Linear Unit）活性化関数。
    元のGPT-2実装で使われている近似式を使用。
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x):
    """
    xの各スコア集合に対してソフトマックス値を計算する。
    最大値を引くことで数値安定性を確保。
    """
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def attention(q, k, v, mask=None):
    """
    スケールドドットプロダクトアテンション。
    q: クエリ (..., seq_len, head_size)
    k: キー (..., seq_len, head_size)
    v: バリュー (..., seq_len, head_size)
    mask: 因果マスク (seq_len, seq_len)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e10, scores)

    probs = softmax(scores)
    return probs @ v

class TransformerBlock:
    """
    GPT-2 トランスフォーマーブロック。
    """
    def __init__(self, params: BlockParams, n_head):
        self.ln_1 = params.ln_1
        self.attn = params.attn
        self.attn.n_head = n_head
        self.ln_2 = params.ln_2
        self.mlp = params.mlp

    def __call__(self, x, kv_cache=_no_cache):
        if kv_cache is not _no_cache:
            attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, new_kv_cache
        # アテンション + 残差接続（Pre-LayerNorm）
        x = x + self.attn(self.ln_1(x))
        # MLP + 残差接続（Pre-LayerNorm）
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2:
    """
    GPT-2 モデル。
    """
    def __init__(self, params: GPT2Params, n_head):
        self.wte = params.wte
        self.wpe = params.wpe
        self.ln_f = params.ln_f
        self.blocks = [TransformerBlock(p, n_head) for p in params.blocks]

    def __call__(self, input_ids, kv_cache=_no_cache):
        # input_ids: (seq_len,)
        seq_len = len(input_ids)

        # 位置埋め込みのオフセット（KV キャッシュ使用時）
        if kv_cache is not _no_cache and kv_cache is not None:
            past_len = kv_cache[0][0].shape[1]
        else:
            past_len = 0
        positions = np.arange(past_len, past_len + seq_len)

        # トークン埋め込み + 位置埋め込み
        x = self.wte[input_ids] + self.wpe[positions]

        if kv_cache is not _no_cache:
            new_kv_cache = []
            for i, block in enumerate(self.blocks):
                layer_cache = kv_cache[i] if kv_cache is not None else None
                x, layer_cache = block(x, kv_cache=layer_cache)
                new_kv_cache.append(layer_cache)
            x = self.ln_f(x)
            return x @ self.wte.T, new_kv_cache

        # トランスフォーマーブロック
        for block in self.blocks:
            x = block(x)

        # 最終LayerNorm
        x = self.ln_f(x)

        # 言語モデルヘッド（重み共有）
        return x @ self.wte.T

def main():
    print("GPT-2の基本関数が実装されています。")

if __name__ == "__main__":
    main()
