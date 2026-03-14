import numpy as np
import pytest
from my_gpt2.model import TransformerBlock, LayerNormParams, AttentionParams, MLPParams, BlockParams

def test_transformer_block_shape():
    batch_size, seq_len, embed_dim = 1, 10, 768
    n_head = 12

    params = BlockParams(
        ln_1=LayerNormParams(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
        attn=AttentionParams(
            w_qkv=np.random.randn(embed_dim, 3 * embed_dim),
            b_qkv=np.zeros(3 * embed_dim),
            w_out=np.random.randn(embed_dim, embed_dim),
            b_out=np.zeros(embed_dim),
        ),
        ln_2=LayerNormParams(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
        mlp=MLPParams(
            w_fc=np.random.randn(embed_dim, 4 * embed_dim),
            b_fc=np.zeros(4 * embed_dim),
            w_proj=np.random.randn(4 * embed_dim, embed_dim),
            b_proj=np.zeros(embed_dim),
        ),
    )

    block = TransformerBlock(params, n_head)
    x = np.random.randn(batch_size, seq_len, embed_dim)

    out = block(x)

    assert out.shape == (batch_size, seq_len, embed_dim)
