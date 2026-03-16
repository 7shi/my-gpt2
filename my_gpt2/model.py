import numpy as np
from dataclasses import dataclass

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

    def __call__(self, x, n_head=None):
        n_head = n_head or self.n_head
        batch_size, seq_len, embed_dim = x.shape
        qkv = np.matmul(x, self.w_qkv) + self.b_qkv

        q, k, v = np.split(qkv, 3, axis=-1)
        head_size = embed_dim // n_head

        def split_heads(tensor):
            return tensor.reshape(batch_size, seq_len, n_head, head_size).transpose(0, 2, 1, 3)

        q, k, v = map(split_heads, [q, k, v])
        mask = np.tril(np.ones((seq_len, seq_len)))
        out = attention(q, k, v, mask=mask)

        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        return np.matmul(out, self.w_out) + self.b_out

@dataclass
class MLPParams:
    w_fc: np.ndarray
    b_fc: np.ndarray
    w_proj: np.ndarray
    b_proj: np.ndarray

    def __call__(self, x):
        a = gelu(np.matmul(x, self.w_fc) + self.b_fc)
        return np.matmul(a, self.w_proj) + self.b_proj

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
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

    if mask is not None:
        scores = np.where(mask == 0, -1e10, scores)

    probs = softmax(scores)
    return np.matmul(probs, v)

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

    def __call__(self, x):
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
        self.params = params
        self.ln_f = params.ln_f
        self.blocks = [TransformerBlock(p, n_head) for p in params.blocks]

    def __call__(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # トークン埋め込み + 位置埋め込み
        # wte: (vocab_size, embed_dim), wpe: (max_pos, embed_dim)
        x = self.params.wte[input_ids] + self.params.wpe[np.arange(input_ids.shape[1])]

        # トランスフォーマーブロック
        for block in self.blocks:
            x = block(x)

        # 最終LayerNorm
        x = self.ln_f(x)

        # 言語モデルヘッド（重み共有）
        # 語彙サイズに射影: (batch_size, seq_len, vocab_size)
        return np.matmul(x, self.params.wte.T)

def main():
    print("GPT-2の基本関数が実装されています。")

if __name__ == "__main__":
    main()
