import numpy as np
import pytest
from my_gpt2.model import MLP

def test_mlp_shape():
    batch_size, seq_len, embed_dim = 1, 10, 768
    x = np.random.randn(batch_size, seq_len, embed_dim)

    params = MLP(
        w_fc=np.random.randn(embed_dim, 4 * embed_dim),
        b_fc=np.zeros(4 * embed_dim),
        w_proj=np.random.randn(4 * embed_dim, embed_dim),
        b_proj=np.zeros(embed_dim),
    )

    out = params(x)

    assert out.shape == (batch_size, seq_len, embed_dim)

def test_mlp_independence():
    # MLPでは各トークンが独立して処理される。
    # 2番目のトークンを変更しても1番目のトークンの出力に影響しないはず。
    batch_size, seq_len, embed_dim = 1, 2, 8
    x1 = np.random.randn(batch_size, seq_len, embed_dim)

    params = MLP(
        w_fc=np.random.randn(embed_dim, 4 * embed_dim),
        b_fc=np.zeros(4 * embed_dim),
        w_proj=np.random.randn(4 * embed_dim, embed_dim),
        b_proj=np.zeros(embed_dim),
    )

    out1 = params(x1)

    x2 = x1.copy()
    x2[:, 1, :] += 10.0 # 2番目のトークンを変更

    out2 = params(x2)

    # 1番目のトークンの出力は同一のはず
    assert np.allclose(out1[:, 0, :], out2[:, 0, :])
