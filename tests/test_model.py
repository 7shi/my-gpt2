import numpy as np
import pytest
from my_gpt2.model import GPT2, LayerNormParams, AttentionParams, MLPParams, BlockParams, GPT2Params

def test_gpt2_full_shape():
    batch_size, seq_len, vocab_size = 1, 5, 100
    embed_dim, n_head, n_layer = 16, 2, 2
    max_pos = 128

    # GPT-2のモックパラメータ
    params = GPT2Params(
        wte=np.random.randn(vocab_size, embed_dim),
        wpe=np.random.randn(max_pos, embed_dim),
        blocks=[
            BlockParams(
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
            for _ in range(n_layer)
        ],
        ln_f=LayerNormParams(g=np.ones(embed_dim), b=np.zeros(embed_dim)),
    )

    model = GPT2(params, n_head)

    # 入力トークンID（整数）
    input_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
