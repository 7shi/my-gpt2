import torch
from dataclasses import dataclass

_no_cache = object()

@dataclass
class LayerNorm:
    g: torch.Tensor
    b: torch.Tensor

    def __call__(self, x, eps=1e-5):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, correction=0)
        return self.g * (x - mean) / torch.sqrt(variance + eps) + self.b

@dataclass
class Attention:
    n_head: int
    w_qkv: torch.Tensor
    b_qkv: torch.Tensor
    w_out: torch.Tensor
    b_out: torch.Tensor

    def __call__(self, x, kv_cache=_no_cache):
        seq_len, embed_dim = x.shape
        qkv = x @ self.w_qkv + self.b_qkv

        q, k, v = torch.chunk(qkv, 3, dim=-1)
        head_size = embed_dim // self.n_head

        def split_heads(tensor):
            tensor = tensor.reshape(seq_len, self.n_head, head_size)
            return tensor.permute(1, 0, 2)

        q, k, v = map(split_heads, [q, k, v])

        if kv_cache is not _no_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=1)
                v = torch.cat([kv_cache[1], v], dim=1)
            kv_len = k.shape[1]
            mask = torch.tril(torch.ones(kv_len, kv_len))[-seq_len:]
            out = attention(q, k, v, mask=mask)
            out = out.permute(1, 0, 2).reshape(seq_len, embed_dim)
            return out @ self.w_out + self.b_out, (k, v)

        mask = torch.tril(torch.ones(seq_len, seq_len))
        out = attention(q, k, v, mask=mask)

        out = out.permute(1, 0, 2).reshape(seq_len, embed_dim)
        return out @ self.w_out + self.b_out

@dataclass
class MLP:
    w_fc: torch.Tensor
    b_fc: torch.Tensor
    w_proj: torch.Tensor
    b_proj: torch.Tensor

    def __call__(self, x):
        a = gelu(x @ self.w_fc + self.b_fc)
        return a @ self.w_proj + self.b_proj

@dataclass
class TransformerBlock:
    ln_1: LayerNorm
    attn: Attention
    ln_2: LayerNorm
    mlp: MLP

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

@dataclass
class GPT2:
    config: dict
    wte: torch.Tensor
    wpe: torch.Tensor
    ln_f: LayerNorm
    blocks: list[TransformerBlock]

    def __call__(self, input_ids, kv_cache=_no_cache):
        # input_ids: (seq_len,)
        seq_len = len(input_ids)

        # 位置埋め込みのオフセット（KV キャッシュ使用時）
        if kv_cache is not _no_cache and kv_cache is not None:
            past_len = kv_cache[0][0].shape[1]
        else:
            past_len = 0
        positions = torch.arange(past_len, past_len + seq_len)

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

def gelu(x):
    """
    GELU（Gaussian Error Linear Unit）活性化関数。
    元のGPT-2実装で使われている近似式を使用。
    """
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3))))

def softmax(x):
    """
    xの各スコア集合に対してソフトマックス値を計算する。
    最大値を引くことで数値安定性を確保。
    """
    exp_x = torch.exp(x - x.max(dim=-1, keepdim=True).values)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

def attention(q, k, v, mask=None):
    """
    スケールドドットプロダクトアテンション。
    q: クエリ (..., seq_len, head_size)
    k: キー (..., seq_len, head_size)
    v: バリュー (..., seq_len, head_size)
    mask: 因果マスク (seq_len, seq_len)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)

    if mask is not None:
        scores = torch.where(mask == 0, torch.tensor(-1e10), scores)

    probs = softmax(scores)
    return probs @ v
