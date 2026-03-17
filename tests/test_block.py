import numpy as np
import pytest
from my_gpt2.model import TransformerBlock, LayerNorm, Attention, MLP

def test_transformer_block_shape():
    batch_size, seq_len, embed_dim = 1, 10, 768
    n_head = 12

    block = TransformerBlock(
        ln_1=LayerNorm(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
        attn=Attention(
            n_head=n_head,
            w_qkv=np.random.randn(embed_dim, 3 * embed_dim),
            b_qkv=np.zeros(3 * embed_dim),
            w_out=np.random.randn(embed_dim, embed_dim),
            b_out=np.zeros(embed_dim),
        ),
        ln_2=LayerNorm(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
        mlp=MLP(
            w_fc=np.random.randn(embed_dim, 4 * embed_dim),
            b_fc=np.zeros(4 * embed_dim),
            w_proj=np.random.randn(4 * embed_dim, embed_dim),
            b_proj=np.zeros(embed_dim),
        ),
    )

    x = np.random.randn(seq_len, embed_dim)
    out = block(x)

    assert out.shape == (seq_len, embed_dim)
